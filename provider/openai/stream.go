package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	_ = stream.Close()
	if contextErr != nil {
		return llmux.Part{}, contextErr
	}
	providerError := &llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
}

type toolBuilder struct {
	id        string
	name      string
	arguments strings.Builder
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
}

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
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return stream.fail(fmt.Errorf("invalid chat stream JSON: %w", err))
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
				if builder == nil {
					builder = &toolBuilder{}
					stream.tools[delta.Index] = builder
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: delta.ID, ToolName: delta.Function.Name})
				}
				if delta.ID != "" {
					builder.id = delta.ID
				}
				builder.name += delta.Function.Name
				if delta.Function.Arguments != "" {
					builder.arguments.WriteString(delta.Function.Arguments)
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
	if stream.textStarted {
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: "text-0"})
	}
	if stream.reasoningStarted {
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: "reasoning-0"})
	}
	for index := 0; index < len(stream.tools); index++ {
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
		call := &llmux.ToolCall{ID: builder.id, Name: builder.name, Arguments: arguments}
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: builder.id}, llmux.Part{Kind: llmux.PartToolCall, ID: builder.id, ToolCall: call})
	}
	if stream.finish == "" {
		stream.finish = llmux.FinishUnknown
	}
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
	tools            map[string]*toolBuilder
	emitted          map[string]bool
	usage            llmux.Usage
	providerState    json.RawMessage
	finish           llmux.FinishReason
	rawFinish        string
	textStarted      bool
	reasoningStarted bool
}

func newResponsesStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, provider string, metadata llmux.ResponseMetadata, includeRaw bool, warnings []string) *responsesStream {
	return &responsesStream{streamBase: streamBase{ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), provider: provider, metadata: metadata, includeRaw: includeRaw, warnings: warnings}, tools: make(map[string]*toolBuilder), emitted: make(map[string]bool)}
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
			Text        string          `json:"text"`
			Name        string          `json:"name"`
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
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return stream.fail(fmt.Errorf("invalid Responses stream JSON: %w", err))
		}
		if event.Type == "" {
			event.Type = frame.Name
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
		}
		if err := stream.mapEvent(event.Type, event.Delta, event.Name, event.ItemID, event.CallID, event.OutputIndex, event.Arguments, event.Item, event.Response, []byte(data)); err != nil {
			return stream.fail(err)
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *responsesStream) mapEvent(eventType, delta, name, itemID, callID string, outputIndex *int, arguments, itemRaw, responseRaw, raw []byte) error {
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
		if !stream.textStarted {
			stream.textStarted = true
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: "text-0"})
		}
		if delta != "" {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: "text-0", Delta: delta})
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
		var item responseItem
		if json.Unmarshal(itemRaw, &item) == nil && isToolItem(item.Type) {
			stream.updateTool(item, outputIndex, false)
		}
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta":
		builder := stream.tool(itemID, callID, outputIndex)
		if name != "" {
			builder.name = name
		}
		builder.arguments.WriteString(delta)
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: first(builder.id, callID, itemID), Delta: delta})
	case "response.function_call_arguments.done", "response.custom_tool_call_input.done":
		builder := stream.tool(itemID, callID, outputIndex)
		if name != "" {
			builder.name = name
		}
		if len(arguments) > 0 {
			argumentText, err := decodeArguments(arguments)
			if err != nil {
				return err
			}
			builder.arguments.Reset()
			builder.arguments.WriteString(argumentText)
		}
		return stream.emitTool(builder)
	case "response.output_item.done":
		var item responseItem
		if json.Unmarshal(itemRaw, &item) == nil && isToolItem(item.Type) {
			builder := stream.updateTool(item, outputIndex, true)
			return stream.emitTool(builder)
		}
	case "response.completed":
		var response responsesResponse
		if err := json.Unmarshal(responseRaw, &response); err != nil {
			return errors.New("invalid response.completed payload")
		}
		stream.providerState = mustMarshal(response.Output)
		stream.usage = mapResponsesUsage(response.Usage)
		stream.finish, stream.rawFinish = responsesFinish(response)
		if stream.textStarted {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: "text-0"})
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

type responseItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func isToolItem(kind string) bool { return kind == "function_call" || kind == "custom_tool_call" }

func (stream *responsesStream) tool(itemID, callID string, index *int) *toolBuilder {
	key := first(callID, itemID)
	if key == "" && index != nil {
		key = fmt.Sprintf("index:%d", *index)
	}
	if builder := stream.tools[key]; builder != nil {
		return builder
	}
	builder := &toolBuilder{id: first(callID, itemID)}
	stream.tools[key] = builder
	stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: builder.id})
	return builder
}

func (stream *responsesStream) updateTool(item responseItem, index *int, replace bool) *toolBuilder {
	builder := stream.tool(item.ID, item.CallID, index)
	if item.CallID != "" {
		builder.id = item.CallID
	} else if builder.id == "" {
		builder.id = item.ID
	}
	if item.Name != "" {
		builder.name = item.Name
	}
	if replace && len(item.Arguments) > 0 {
		if text, err := decodeArguments(item.Arguments); err == nil {
			builder.arguments.Reset()
			builder.arguments.WriteString(text)
		}
	}
	return builder
}

func (stream *responsesStream) emitTool(builder *toolBuilder) error {
	id := builder.id
	if stream.emitted[id] {
		return nil
	}
	arguments := json.RawMessage(builder.arguments.String())
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	if !json.Valid(arguments) {
		return fmt.Errorf("Responses tool call %q has invalid JSON arguments", id)
	}
	stream.emitted[id] = true
	call := &llmux.ToolCall{ID: id, Name: builder.name, Arguments: arguments}
	stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: id}, llmux.Part{Kind: llmux.PartToolCall, ID: id, ToolCall: call})
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
