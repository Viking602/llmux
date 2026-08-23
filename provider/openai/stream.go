package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/Viking602/llmux"
	internalstream "github.com/Viking602/llmux/internal/stream"
)

type streamBase struct {
	ctx        context.Context
	cancel     context.CancelFunc
	body       io.ReadCloser
	reader     *internalstream.SSEReader
	provider   string
	metadata   llmux.ResponseMetadata
	includeRaw bool
	warnings   []string
	pending    []llmux.Part
	terminal   bool
	closeOnce  sync.Once
}

func (stream *streamBase) Close() error {
	var err error
	stream.closeOnce.Do(func() {
		stream.cancel()
		err = stream.body.Close()
	})
	return err
}

func (stream *streamBase) pop() (llmux.Part, bool) {
	if len(stream.pending) == 0 {
		return llmux.Part{}, false
	}
	part := stream.pending[0]
	stream.pending = stream.pending[1:]
	return part, true
}

func (stream *streamBase) fail(err error) (llmux.Part, error) {
	contextErr := stream.ctx.Err()
	stream.terminal = true
	stream.pending = nil
	_ = stream.Close()
	if contextErr != nil {
		return llmux.Part{}, contextErr
	}
	providerError := &llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
}

type toolBuilder struct {
	id             string
	callID         string
	identity       string
	kind           string
	name           string
	arguments      strings.Builder
	argumentsDone  bool
	emitted        bool
	emittedID      string
	outputIndex    int
	hasOutputIndex bool
}

type chatStream struct {
	streamBase
	tools            map[int]*toolBuilder
	usage            llmux.Usage
	finish           llmux.FinishReason
	rawFinish        string
	profile          CompatProfile
	textStarted      bool
	reasoningStarted bool
	finalized        bool
	toolCalls        internalstream.ToolCallTracker
	toolBudget       internalstream.ToolCallBudget
	activeToolIDs    map[string]int
}

const maxChatFanoutParts = 4096

func newChatStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, provider string, metadata llmux.ResponseMetadata, includeRaw bool, profile CompatProfile, warnings []string) *chatStream {
	return &chatStream{streamBase: streamBase{ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), provider: provider, metadata: metadata, includeRaw: includeRaw, warnings: warnings}, tools: make(map[int]*toolBuilder), profile: profile}
}

func (stream *chatStream) Recv() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	if stream.terminal {
		return llmux.Part{}, io.EOF
	}
	for {
		event, err := stream.reader.Next()
		if err != nil {
			if errors.Is(err, io.EOF) && stream.ctx.Err() == nil {
				return stream.fail(errors.New("chat stream ended before [DONE]"))
			}
			return stream.fail(err)
		}
		data := strings.TrimSpace(event.Data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			stream.finishParts()
			stream.terminal = true
			_ = stream.Close()
			return stream.popOrEOF()
		}
		var chunk struct {
			ID      string `json:"id"`
			Model   string `json:"model"`
			Created int64  `json:"created"`
			Choices []struct {
				Delta struct {
					Content          *string `json:"content"`
					ReasoningContent *string `json:"reasoning_content"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage chatUsage `json:"usage"`
			XGroq struct {
				Usage chatUsage `json:"usage"`
			} `json:"x_groq"`
			Citations []string        `json:"citations"`
			Code      string          `json:"code"`
			Error     json.RawMessage `json:"error"`
		}
		if err := internalstream.ValidateJSONComplexity([]byte(data)); err != nil {
			return stream.fail(err)
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return stream.fail(fmt.Errorf("invalid chat stream JSON: %w", err))
		}
		fanout := len(chunk.Citations) + len(chunk.Choices)
		for _, choice := range chunk.Choices {
			fanout += len(choice.Delta.ToolCalls)
		}
		if fanout > maxChatFanoutParts {
			return stream.fail(fmt.Errorf("chat stream frame expands to %d parts", fanout))
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
		}
		if len(chunk.Error) > 0 && string(chunk.Error) != "null" {
			var message string
			if json.Unmarshal(chunk.Error, &message) != nil {
				message = string(chunk.Error)
			}
			return stream.fail(&llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Code: chunk.Code, Message: message})
		}
		for index, uri := range chunk.Citations {
			source := &llmux.Source{ID: fmt.Sprintf("citation-%d", index), URL: uri}
			content := &llmux.ContentPart{Kind: llmux.ContentSource, Source: source}
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartSource, Content: content})
		}
		if stream.metadata.ID == "" && chunk.ID != "" {
			stream.metadata.ID = chunk.ID
			stream.metadata.ModelID = first(chunk.Model, stream.metadata.ModelID)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartResponseMetadata, Response: stream.metadata})
		}
		if chunk.Usage.TotalTokens > 0 || chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			stream.usage = usageFromChat(chunk.Usage)
		}
		if stream.profile.StreamUsageKey == "x_groq" && (chunk.XGroq.Usage.TotalTokens > 0 || chunk.XGroq.Usage.PromptTokens > 0 || chunk.XGroq.Usage.CompletionTokens > 0) {
			stream.usage = usageFromChat(chunk.XGroq.Usage)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != nil {
				if !stream.textStarted {
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: "text-0"})
					stream.textStarted = true
				}
				if *choice.Delta.Content != "" {
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: "text-0", Delta: *choice.Delta.Content})
				}
			}
			if choice.Delta.ReasoningContent != nil {
				if !stream.reasoningStarted {
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningStart, ID: "reasoning-0"})
					stream.reasoningStarted = true
				}
				if *choice.Delta.ReasoningContent != "" {
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: "reasoning-0", Delta: *choice.Delta.ReasoningContent})
				}
			}
			for _, delta := range choice.Delta.ToolCalls {
				builder := stream.tools[delta.Index]
				created := builder == nil
				if created {
					if err := stream.toolBudget.Begin(); err != nil {
						return stream.fail(fmt.Errorf("chat %w", err))
					}
					builder = &toolBuilder{}
					stream.tools[delta.Index] = builder
				}
				if delta.ID != "" {
					if builder.id != "" && builder.id != delta.ID {
						return stream.fail(fmt.Errorf("chat tool index %d changed call ID from %q to %q", delta.Index, builder.id, delta.ID))
					}
					if activeIndex, exists := stream.activeToolIDs[delta.ID]; exists && activeIndex != delta.Index {
						return stream.fail(fmt.Errorf("chat tool call ID %q is already active at index %d", delta.ID, activeIndex))
					}
					if builder.id == "" {
						if err := stream.toolBudget.AddAlias(delta.ID); err != nil {
							return stream.fail(fmt.Errorf("chat %w", err))
						}
						if stream.activeToolIDs == nil {
							stream.activeToolIDs = make(map[string]int)
						}
						stream.activeToolIDs[delta.ID] = delta.Index
					}
					builder.id = delta.ID
				}
				if delta.Function.Name != "" {
					if len(builder.name)+len(delta.Function.Name) > internalstream.MaxToolMetadataBytes {
						return stream.fail(fmt.Errorf("chat tool name exceeds %d bytes", internalstream.MaxToolMetadataBytes))
					}
					if err := stream.toolBudget.AddMetadata(delta.Function.Name); err != nil {
						return stream.fail(fmt.Errorf("chat %w", err))
					}
					builder.name += delta.Function.Name
				}
				if created {
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: builder.id, ToolName: builder.name})
				}
				if delta.Function.Arguments != "" {
					if err := stream.toolBudget.AppendArguments(&builder.arguments, delta.Function.Arguments); err != nil {
						return stream.fail(fmt.Errorf("chat %w", err))
					}
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: builder.id, Delta: delta.Function.Arguments})
				}
			}
			if choice.FinishReason != nil {
				stream.rawFinish = *choice.FinishReason
				stream.finish = finishReason(*choice.FinishReason)
			}
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *chatStream) finishParts() {
	if stream.finalized {
		return
	}
	stream.finalized = true
	if stream.textStarted {
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: "text-0"})
	}
	if stream.reasoningStarted {
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: "reasoning-0"})
	}
	indexes := make([]int, 0, len(stream.tools))
	for index := range stream.tools {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		builder := stream.tools[index]
		if builder == nil {
			continue
		}
		arguments := json.RawMessage(builder.arguments.String())
		if len(arguments) == 0 {
			arguments = json.RawMessage(`{}`)
		}
		if !json.Valid(arguments) {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartError, Err: &llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Message: "tool call has invalid JSON arguments"}})
			continue
		}
		accepted, err := stream.toolCalls.Accept(builder.id, builder.name, arguments)
		if err != nil {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartError, Err: &llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Message: err.Error()}})
			continue
		}
		if !accepted {
			continue
		}
		call := &llmux.ToolCall{ID: builder.id, Name: builder.name, Arguments: arguments}
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: builder.id}, llmux.Part{Kind: llmux.PartToolCall, ID: builder.id, ToolCall: call})
	}
	if stream.finish == "" {
		stream.finish = llmux.FinishUnknown
	}
	stream.usage = normalizeProfileUsage(stream.usage, stream.profile)
	stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFinish, FinishReason: stream.finish, RawFinishReason: stream.rawFinish, Usage: stream.usage, Warnings: stream.warnings})
}

func (stream *chatStream) popOrEOF() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	return llmux.Part{}, io.EOF
}

type responsesStream struct {
	streamBase
	tools                 []*toolBuilder
	toolsByOutputIndex    map[int]*toolBuilder
	toolsByItemID         map[string]*toolBuilder
	toolsByCallID         map[string]*toolBuilder
	toolsByPrefixedCallID map[string]*toolBuilder
	toolArguments         internalstream.ToolCallTracker
	toolCalls             internalstream.ToolCallTracker
	profile               CompatProfile
	toolBudget            internalstream.ToolCallBudget
	usage                 llmux.Usage
	completed             bool
	providerState         json.RawMessage
	finish                llmux.FinishReason
	rawFinish             string
	textStarted           map[int]bool
	textPhases            map[int]llmux.TextPhase
	reasoningStarted      bool
}

func newResponsesStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, provider string, metadata llmux.ResponseMetadata, includeRaw bool, profile CompatProfile, warnings []string) *responsesStream {
	return &responsesStream{
		streamBase:            streamBase{ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), provider: provider, metadata: metadata, includeRaw: includeRaw, warnings: warnings},
		toolsByOutputIndex:    make(map[int]*toolBuilder),
		toolsByItemID:         make(map[string]*toolBuilder),
		profile:               profile,
		toolsByCallID:         make(map[string]*toolBuilder),
		toolsByPrefixedCallID: make(map[string]*toolBuilder),
		textStarted:           make(map[int]bool),
		textPhases:            make(map[int]llmux.TextPhase),
	}
}

func (stream *responsesStream) Recv() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	if stream.terminal {
		return llmux.Part{}, io.EOF
	}
	for {
		frame, err := stream.reader.Next()
		if err != nil {
			if errors.Is(err, io.EOF) && stream.ctx.Err() == nil {
				if stream.finish != "" {
					stream.terminal = true
					_ = stream.Close()
					return llmux.Part{}, io.EOF
				}
				return stream.fail(errors.New("Responses stream ended before response.completed"))
			}
			return stream.fail(err)
		}
		data := strings.TrimSpace(frame.Data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			if stream.finish == "" {
				return stream.fail(errors.New("Responses stream ended before response.completed"))
			}
			stream.terminal = true
			_ = stream.Close()
			return llmux.Part{}, io.EOF
		}
		var event struct {
			Type        string          `json:"type"`
			Delta       string          `json:"delta"`
			Phase       llmux.TextPhase `json:"phase"`
			Text        string          `json:"text"`
			Name        string          `json:"name"`
			Input       string          `json:"input"`
			Arguments   json.RawMessage `json:"arguments"`
			ItemID      string          `json:"item_id"`
			CallID      string          `json:"call_id"`
			OutputIndex *int            `json:"output_index"`
			Item        json.RawMessage `json:"item"`
			Response    json.RawMessage `json:"response"`
			Error       json.RawMessage `json:"error"`
			Code        string          `json:"code"`
			Message     string          `json:"message"`
		}
		if err := internalstream.ValidateJSONComplexity([]byte(data)); err != nil {
			return stream.fail(err)
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return stream.fail(fmt.Errorf("invalid Responses stream JSON: %w", err))
		}
		if event.Type == "" {
			event.Type = frame.Name
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
		}
		if err := stream.mapEvent(event.Type, event.Delta, event.Phase, event.Name, event.Input, event.ItemID, event.CallID, event.OutputIndex, event.Arguments, event.Item, event.Response, []byte(data)); err != nil {
			return stream.fail(err)
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *responsesStream) mapEvent(eventType, delta string, phase llmux.TextPhase, name, input, itemID, callID string, outputIndex *int, arguments, itemRaw, responseRaw, raw []byte) error {
	if stream.completed && eventType != "response.completed" && isResponsesToolLifecycleEvent(eventType) {
		return fmt.Errorf("Responses tool event %q arrived after response.completed", eventType)
	}
	switch eventType {
	case "response.created", "response.in_progress":
		if len(responseRaw) > 0 && stream.metadata.ID == "" {
			var response struct {
				ID    string `json:"id"`
				Model string `json:"model"`
			}
			if json.Unmarshal(responseRaw, &response) == nil {
				stream.metadata.ID = response.ID
				stream.metadata.ModelID = first(response.Model, stream.metadata.ModelID)
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartResponseMetadata, Response: stream.metadata})
			}
		}
	case "response.output_text.delta":
		index := valueOrZero(outputIndex)
		if normalized := normalizeTextPhase(phase); normalized != "" {
			stream.textPhases[index] = normalized
		}
		textPhase := stream.textPhases[index]
		id := fmt.Sprintf("text-%d", index)
		if !stream.textStarted[index] {
			stream.textStarted[index] = true
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: id, TextPhase: textPhase})
		}
		if delta != "" {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: id, Delta: delta, TextPhase: textPhase})
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if !stream.reasoningStarted {
			stream.reasoningStarted = true
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningStart, ID: "reasoning-0"})
		}
		if delta != "" {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: "reasoning-0", Delta: delta})
		}
	case "response.output_item.added":
		item, toolItem, err := decodeResponseToolItem(itemRaw)
		if err != nil {
			return err
		}
		if toolItem {
			_, err = stream.updateTool(item, outputIndex, false)
			return err
		}
		var textItem struct {
			Phase llmux.TextPhase `json:"phase"`
		}
		if json.Unmarshal(itemRaw, &textItem) == nil && outputIndex != nil {
			if normalized := normalizeTextPhase(textItem.Phase); normalized != "" {
				stream.textPhases[*outputIndex] = normalized
			}
		}
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta":
		builder, err := stream.tool(itemID, callID, outputIndex)
		if err != nil {
			return err
		}
		eventKind := "function_call"
		if eventType == "response.custom_tool_call_input.delta" {
			eventKind = "custom_tool_call"
		}
		if builder.kind != "" && builder.kind != eventKind {
			return fmt.Errorf("Responses tool call identity %q changed kind from %q to %q", builder.identity, builder.kind, eventKind)
		}
		builder.kind = eventKind
		if builder.argumentsDone {
			return fmt.Errorf("Responses tool call identity %q received arguments after finalization", builder.identity)
		}
		if err := stream.setToolName(builder, name); err != nil {
			return err
		}
		if err := stream.toolBudget.AppendArguments(&builder.arguments, delta); err != nil {
			return fmt.Errorf("Responses %w", err)
		}
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: first(builder.id, callID, itemID), Delta: delta})
	case "response.function_call_arguments.done":
		builder, err := stream.toolForTerminalEvent(itemID, callID, outputIndex)
		if err != nil {
			return err
		}
		if builder.kind != "" && builder.kind != "function_call" {
			return fmt.Errorf("Responses tool call identity %q changed kind from %q to function_call", builder.identity, builder.kind)
		}
		builder.kind = "function_call"
		argumentText, err := decodeArguments(arguments)
		if err != nil {
			return err
		}
		return stream.finalizeToolArguments(builder, name, json.RawMessage(argumentText))
	case "response.custom_tool_call_input.done":
		builder, err := stream.toolForTerminalEvent(itemID, callID, outputIndex)
		if err != nil {
			return err
		}
		if builder.kind != "" && builder.kind != "custom_tool_call" {
			return fmt.Errorf("Responses tool call identity %q changed kind from %q to custom_tool_call", builder.identity, builder.kind)
		}
		builder.kind = "custom_tool_call"
		return stream.finalizeToolArguments(builder, name, customToolArguments(input))
	case "response.output_item.done":
		item, toolItem, err := decodeResponseToolItem(itemRaw)
		if err != nil {
			return err
		}
		if toolItem {
			if stream.completed {
				existing, err := stream.lookupTool(item.ID, item.CallID, outputIndex)
				if err != nil {
					return err
				}
				if existing == nil {
					return errors.New("Responses output_item.done introduced a tool call after response.completed")
				}
				if !existing.emitted {
					return errors.New("Responses output_item.done finalized an uncommitted tool call after response.completed")
				}
			}
			builder, err := stream.updateTool(item, outputIndex, true)
			if err != nil {
				return err
			}
			if builder.callID == "" {
				return nil
			}
			return stream.emitTool(builder)
		}
	case "response.completed":
		var response responsesResponse
		if err := json.Unmarshal(responseRaw, &response); err != nil {
			return errors.New("invalid response.completed payload")
		}
		providerState := mustMarshal(response.Output)
		builders := make([]*toolBuilder, 0, len(response.Output))
		for index, rawItem := range response.Output {
			item, toolItem, err := decodeResponseToolItem(rawItem)
			if err != nil {
				return err
			}
			if !toolItem {
				continue
			}
			if stream.completed {
				existing, err := stream.lookupTool(item.ID, item.CallID, &index)
				if err != nil {
					return err
				}
				if existing == nil {
					return fmt.Errorf("Responses repeated response.completed introduced a new tool call at output index %d", index)
				}
				if !existing.emitted {
					return fmt.Errorf("Responses repeated response.completed finalized an uncommitted tool call at output index %d", index)
				}
			}
			builder, err := stream.updateTool(item, &index, true)
			if err != nil {
				return err
			}
			builders = append(builders, builder)
		}
		for _, builder := range builders {
			if err := stream.emitTool(builder); err != nil {
				return err
			}
		}
		if stream.completed {
			return nil
		}
		if err := stream.emitPendingTools(); err != nil {
			return err
		}
		stream.completed = true
		stream.providerState = providerState
		stream.usage = normalizeProfileUsage(mapResponsesUsage(response.Usage), stream.profile)
		stream.finish, stream.rawFinish = responsesFinish(response)
		textIndexes := make([]int, 0, len(stream.textStarted))
		for index := range stream.textStarted {
			textIndexes = append(textIndexes, index)
		}
		sort.Ints(textIndexes)
		for _, index := range textIndexes {
			stream.pending = append(stream.pending, llmux.Part{
				Kind: llmux.PartTextEnd, ID: fmt.Sprintf("text-%d", index), TextPhase: stream.textPhases[index],
			})
		}
		if stream.reasoningStarted {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: "reasoning-0"})
		}
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFinish, FinishReason: stream.finish, RawFinishReason: stream.rawFinish, Usage: stream.usage, ProviderState: stream.providerState, Warnings: stream.warnings})
	case "response.failed", "response.incomplete", "error":
		return responsesStreamError(raw)
	}
	return nil
}

func isResponsesToolLifecycleEvent(eventType string) bool {
	switch eventType {
	case "response.output_item.added",
		"response.function_call_arguments.delta",
		"response.custom_tool_call_input.delta":
		return true
	default:
		return false
	}
}

type responseItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Input     *string         `json:"input"`
	Arguments json.RawMessage `json:"arguments"`
}

func isToolItem(kind string) bool { return kind == "function_call" || kind == "custom_tool_call" }

func decodeResponseToolItem(raw json.RawMessage) (responseItem, bool, error) {
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return responseItem{}, false, fmt.Errorf("invalid Responses output item: %w", err)
	}
	if !isToolItem(envelope.Type) {
		return responseItem{}, false, nil
	}
	var item responseItem
	if err := json.Unmarshal(raw, &item); err != nil {
		return responseItem{}, false, fmt.Errorf("invalid Responses %s item: %w", envelope.Type, err)
	}
	return item, true, nil
}

func (stream *responsesStream) tool(itemID, callID string, index *int) (*toolBuilder, error) {
	builder, err := stream.lookupTool(itemID, callID, index)
	if err != nil {
		return nil, err
	}
	if builder == nil {
		if err := stream.toolBudget.Begin(); err != nil {
			return nil, fmt.Errorf("Responses %w", err)
		}
		identity := fmt.Sprintf("responses:%d", len(stream.tools))
		builder = &toolBuilder{id: first(callID, itemID), identity: identity}
		stream.tools = append(stream.tools, builder)
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: builder.id})
	}
	if err := stream.registerToolAliases(builder, itemID, callID, index); err != nil {
		return nil, err
	}
	return builder, nil
}

func (stream *responsesStream) toolForTerminalEvent(itemID, callID string, index *int) (*toolBuilder, error) {
	if !stream.completed {
		return stream.tool(itemID, callID, index)
	}
	builder, err := stream.lookupTool(itemID, callID, index)
	if err != nil {
		return nil, err
	}
	if builder == nil {
		return nil, errors.New("Responses terminal tool event introduced a call after response.completed")
	}
	if !builder.emitted {
		return nil, errors.New("Responses terminal tool event finalized an uncommitted call after response.completed")
	}
	if err := stream.registerToolAliases(builder, itemID, callID, index); err != nil {
		return nil, err
	}
	return builder, nil
}

func (stream *responsesStream) lookupTool(itemID, callID string, index *int) (*toolBuilder, error) {
	var found *toolBuilder
	merge := func(candidate *toolBuilder) error {
		if candidate == nil {
			return nil
		}
		if found != nil && found != candidate {
			return errors.New("Responses tool aliases resolve to different calls")
		}
		found = candidate
		return nil
	}
	if index != nil {
		if err := merge(stream.toolsByOutputIndex[*index]); err != nil {
			return nil, err
		}
	}
	if itemID != "" {
		if err := merge(stream.toolsByItemID[itemID]); err != nil {
			return nil, err
		}
		if err := merge(stream.toolsByPrefixedCallID[itemID]); err != nil {
			return nil, err
		}
		if err := merge(stream.toolsByCallID[itemID]); err != nil {
			return nil, err
		}
	}
	if callID != "" {
		callBuilder := stream.toolsByCallID[callID]
		if err := merge(callBuilder); err != nil {
			return nil, err
		}
		if callBuilder == nil {
			if err := merge(stream.toolsByItemID[callID]); err != nil {
				return nil, err
			}
			if err := merge(stream.toolsByPrefixedCallID["fc_"+callID]); err != nil {
				return nil, err
			}
		}
	}
	return found, nil
}

func (stream *responsesStream) registerToolAliases(builder *toolBuilder, itemID, callID string, index *int) error {
	if callID != "" && builder.callID != "" && builder.callID != callID {
		return fmt.Errorf("Responses tool call identity %q changed call ID from %q to %q", builder.identity, builder.callID, callID)
	}
	if index != nil {
		if builder.hasOutputIndex && builder.outputIndex != *index {
			return fmt.Errorf("Responses tool call identity %q changed output index from %d to %d", builder.identity, builder.outputIndex, *index)
		}
		if existing := stream.toolsByOutputIndex[*index]; existing != nil && existing != builder {
			return fmt.Errorf("Responses output index %d is already bound to another tool call", *index)
		}
		if stream.toolsByOutputIndex[*index] == nil {
			if err := stream.toolBudget.AddAlias("output_index"); err != nil {
				return fmt.Errorf("Responses %w", err)
			}
		}
		stream.toolsByOutputIndex[*index] = builder
		builder.outputIndex = *index
		builder.hasOutputIndex = true
	}
	if err := stream.bindToolAlias(stream.toolsByItemID, itemID, builder, "item ID"); err != nil {
		return err
	}
	if callID == "" && strings.HasPrefix(itemID, "fc_") {
		if err := stream.bindToolAlias(stream.toolsByPrefixedCallID, itemID, builder, "prefixed item ID"); err != nil {
			return err
		}
	}
	if err := stream.bindToolAlias(stream.toolsByCallID, callID, builder, "call ID"); err != nil {
		return err
	}
	if callID != "" && builder.callID == "" {
		if err := stream.bindToolAlias(stream.toolsByPrefixedCallID, "fc_"+callID, builder, "prefixed call ID"); err != nil {
			return err
		}
		builder.callID = callID
	}
	return nil
}

func (stream *responsesStream) bindToolAlias(index map[string]*toolBuilder, alias string, builder *toolBuilder, kind string) error {
	if alias == "" {
		return nil
	}
	if existing := index[alias]; existing != nil && existing != builder {
		return fmt.Errorf("Responses %s %q is already bound to another tool call", kind, alias)
	}
	if index[alias] == nil {
		if err := stream.toolBudget.AddAlias(alias); err != nil {
			return fmt.Errorf("Responses %w", err)
		}
	}
	index[alias] = builder
	return nil
}

func (stream *responsesStream) updateTool(item responseItem, index *int, replace bool) (*toolBuilder, error) {
	builder, err := stream.tool(item.ID, item.CallID, index)
	if err != nil {
		return nil, err
	}
	if item.CallID != "" {
		builder.id = item.CallID
	} else if builder.id == "" {
		builder.id = item.ID
	}
	if item.Type != "" {
		if builder.kind != "" && builder.kind != item.Type {
			return nil, fmt.Errorf("Responses tool call identity %q changed kind from %q to %q", builder.identity, builder.kind, item.Type)
		}
		builder.kind = item.Type
	}
	if err := stream.setToolName(builder, item.Name); err != nil {
		return nil, err
	}
	if replace {
		arguments, present, err := responseItemArguments(item)
		if err != nil {
			return nil, err
		}
		if !present {
			if builder.kind == "custom_tool_call" && !builder.argumentsDone {
				arguments = customToolArguments(builder.arguments.String())
			} else if builder.arguments.Len() > 0 {
				arguments = append(json.RawMessage(nil), builder.arguments.String()...)
			} else {
				arguments = json.RawMessage(`{}`)
			}
		}
		if err := stream.finalizeToolArguments(builder, item.Name, arguments); err != nil {
			return nil, err
		}
	}
	return builder, nil
}

func (stream *responsesStream) setToolName(builder *toolBuilder, name string) error {
	if name == "" {
		return nil
	}
	if builder.name != "" && builder.name != name {
		return fmt.Errorf("Responses tool call identity %q changed name from %q to %q", builder.identity, builder.name, name)
	}
	if builder.name == "" {
		if err := stream.toolBudget.AddMetadata(name); err != nil {
			return fmt.Errorf("Responses %w", err)
		}
		builder.name = name
	}
	return nil
}

func (stream *responsesStream) finalizeToolArguments(builder *toolBuilder, name string, arguments json.RawMessage) error {
	if err := stream.setToolName(builder, name); err != nil {
		return err
	}
	accepted, err := stream.toolArguments.Accept(builder.identity, "<arguments>", arguments)
	if err != nil {
		return fmt.Errorf("Responses %w", err)
	}
	if !accepted {
		return nil
	}
	builder.arguments.Reset()
	builder.arguments.Write(arguments)
	builder.argumentsDone = true
	return nil
}

func responseItemArguments(item responseItem) (json.RawMessage, bool, error) {
	if item.Type == "custom_tool_call" {
		if item.Input == nil {
			return nil, false, nil
		}
		return customToolArguments(*item.Input), true, nil
	}
	if len(item.Arguments) == 0 {
		return nil, false, nil
	}
	text, err := decodeArguments(item.Arguments)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(text), true, nil
}

func customToolArguments(input string) json.RawMessage {
	arguments, _ := json.Marshal(map[string]string{"input": input})
	return arguments
}

func (stream *responsesStream) emitPendingTools() error {
	for _, builder := range stream.tools {
		if builder.emitted || (!builder.argumentsDone && builder.arguments.Len() == 0) {
			continue
		}
		if builder.kind == "custom_tool_call" && !builder.argumentsDone {
			if err := stream.finalizeToolArguments(builder, builder.name, customToolArguments(builder.arguments.String())); err != nil {
				return err
			}
		}
		if err := stream.emitTool(builder); err != nil {
			return err
		}
	}
	return nil
}

func (stream *responsesStream) emitTool(builder *toolBuilder) error {
	id := builder.id
	arguments := json.RawMessage(builder.arguments.String())
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	if builder.emitted && builder.emittedID != id {
		return fmt.Errorf("Responses tool call identity %q changed final ID from %q to %q", builder.identity, builder.emittedID, id)
	}
	if !json.Valid(arguments) {
		return fmt.Errorf("Responses tool call %q has invalid JSON arguments", id)
	}
	accepted, err := stream.toolCalls.Accept(builder.identity, builder.name, arguments)
	if err != nil {
		return fmt.Errorf("Responses %w", err)
	}
	if !accepted {
		return nil
	}
	builder.emitted = true
	builder.emittedID = id
	call := &llmux.ToolCall{ID: id, Name: builder.name, Arguments: arguments}
	stream.pending = append(stream.pending,
		llmux.Part{Kind: llmux.PartToolInputEnd, ID: id},
		llmux.Part{Kind: llmux.PartToolCall, ID: id, ToolCall: call},
	)
	return nil
}

func decodeArguments(raw json.RawMessage) (string, error) {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		if !json.Valid([]byte(text)) {
			return "", errors.New("tool arguments are not valid JSON")
		}
		return text, nil
	}
	if json.Valid(raw) {
		return string(raw), nil
	}
	return "", errors.New("tool arguments are not valid JSON")
}

func responsesStreamError(raw []byte) error {
	var envelope struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Error   *struct {
			Code    string `json:"code"`
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		Response *struct {
			Error *struct {
				Code    string `json:"code"`
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		} `json:"response"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return errors.New("malformed Responses error event")
	}
	failure := envelope.Error
	if failure == nil && envelope.Response != nil {
		failure = envelope.Response.Error
	}
	if failure != nil {
		return &llmux.ProviderError{Kind: llmux.ErrorStream, Code: first(failure.Code, failure.Type), Message: failure.Message}
	}
	return &llmux.ProviderError{Kind: llmux.ErrorStream, Code: envelope.Code, Message: first(envelope.Message, "Responses stream failed")}
}

func valueOrZero(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

func normalizeTextPhase(phase llmux.TextPhase) llmux.TextPhase {
	switch phase {
	case llmux.TextPhaseCommentary, llmux.TextPhaseFinalAnswer:
		return phase
	default:
		return llmux.TextPhaseUnspecified
	}
}
