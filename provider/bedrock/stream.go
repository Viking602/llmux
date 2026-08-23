package bedrock

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

type blockBuilder struct {
	kind          string
	id            string
	name          string
	text          strings.Builder
	signature     strings.Builder
	input         strings.Builder
	suppressInput bool
	finalized     bool
}

type converseStream struct {
	ctx           context.Context
	cancel        context.CancelFunc
	body          io.ReadCloser
	warnings      []string
	includeRaw    bool
	blocks        map[int]*blockBuilder
	pending       []llmux.Part
	usage         llmux.Usage
	finish        llmux.FinishReason
	rawFinish     string
	stopped       bool
	terminal      bool
	closeOnce     sync.Once
	toolCalls     internalstream.ToolCallTracker
	toolBudget    internalstream.ToolCallBudget
	stateBudget   internalstream.StateBudget
	activeToolIDs map[string]int
}

func newConverseStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, warnings []string, includeRaw bool) *converseStream {
	return &converseStream{
		ctx: ctx, cancel: cancel, body: body, warnings: warnings, includeRaw: includeRaw,
		blocks: make(map[int]*blockBuilder), activeToolIDs: make(map[string]int),
	}
}

func (stream *converseStream) Close() error {
	var err error
	stream.closeOnce.Do(func() { stream.cancel(); err = stream.body.Close() })
	return err
}

func (stream *converseStream) Recv() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	if stream.terminal {
		return llmux.Part{}, io.EOF
	}
	for {
		event, err := readEvent(stream.body)
		if err != nil {
			if errors.Is(err, io.EOF) && stream.stopped && stream.ctx.Err() == nil {
				stream.finishPart()
				stream.terminal = true
				_ = stream.Close()
				return stream.popOrEOF()
			}
			return stream.fail(err)
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: append([]byte(nil), event.Payload...)})
		}
		if event.MessageType == "exception" || strings.HasSuffix(event.EventType, "Exception") {
			var failure struct {
				Message string `json:"message"`
			}
			_ = json.Unmarshal(event.Payload, &failure)
			return stream.fail(&llmux.ProviderError{Provider: "amazon-bedrock", Kind: llmux.ErrorStream, Code: event.EventType, Message: first(failure.Message, string(event.Payload))})
		}
		if err := stream.mapEvent(event); err != nil {
			return stream.fail(err)
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *converseStream) mapEvent(event eventMessage) error {
	if stream.stopped && event.EventType != "messageStop" && event.EventType != "metadata" {
		return fmt.Errorf("Bedrock event %q arrived after messageStop", event.EventType)
	}
	if len(event.Payload) > 0 {
		if err := internalstream.ValidateJSONComplexity(event.Payload); err != nil {
			return err
		}
	}
	switch event.EventType {
	case "messageStart":
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartResponseMetadata, Response: llmux.ResponseMetadata{}})
	case "contentBlockStart":
		var payload struct {
			ContentBlockIndex int `json:"contentBlockIndex"`
			Start             struct {
				ToolUse *struct {
					ToolUseID string `json:"toolUseId"`
					Name      string `json:"name"`
				} `json:"toolUse"`
			} `json:"start"`
		}
		if err := json.Unmarshal(event.Payload, &payload); err != nil {
			return err
		}
		if stream.blocks[payload.ContentBlockIndex] != nil {
			return fmt.Errorf("Bedrock content block index %d was started more than once", payload.ContentBlockIndex)
		}
		if payload.Start.ToolUse != nil {
			if err := stream.toolBudget.Begin(); err != nil {
				return fmt.Errorf("Bedrock %w", err)
			}
			builder := &blockBuilder{kind: "toolUse", id: payload.Start.ToolUse.ToolUseID, name: payload.Start.ToolUse.Name}
			if err := stream.toolBudget.AddAlias(builder.id); err != nil {
				return fmt.Errorf("Bedrock %w", err)
			}
			if err := stream.toolBudget.AddMetadata(builder.name); err != nil {
				return fmt.Errorf("Bedrock %w", err)
			}
			if builder.id != "" {
				if stream.activeToolIDs == nil {
					stream.activeToolIDs = make(map[string]int)
				}
				if activeIndex, exists := stream.activeToolIDs[builder.id]; exists {
					return fmt.Errorf("Bedrock tool call ID %q is already active at block %d", builder.id, activeIndex)
				}
				stream.activeToolIDs[builder.id] = payload.ContentBlockIndex
			}
			builder.suppressInput = stream.toolCalls.Seen(builder.id)
			stream.blocks[payload.ContentBlockIndex] = builder
			if !builder.suppressInput {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: builder.id, ToolName: builder.name})
			}
		}
	case "contentBlockDelta":
		var payload struct {
			ContentBlockIndex int `json:"contentBlockIndex"`
			Delta             struct {
				Text    *string `json:"text"`
				ToolUse *struct {
					Input string `json:"input"`
				} `json:"toolUse"`
				Reasoning *struct {
					Text      string `json:"text"`
					Signature string `json:"signature"`
				} `json:"reasoningContent"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(event.Payload, &payload); err != nil {
			return err
		}
		builder := stream.blocks[payload.ContentBlockIndex]
		if payload.Delta.Text != nil {
			if builder == nil {
				builder = &blockBuilder{kind: "text"}
				stream.blocks[payload.ContentBlockIndex] = builder
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: indexID(payload.ContentBlockIndex)})
			}
			builder.text.WriteString(*payload.Delta.Text)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: indexID(payload.ContentBlockIndex), Delta: *payload.Delta.Text})
		}
		if payload.Delta.Reasoning != nil {
			if builder == nil {
				builder = &blockBuilder{kind: "reasoning"}
				stream.blocks[payload.ContentBlockIndex] = builder
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningStart, ID: indexID(payload.ContentBlockIndex)})
			}
			builder.text.WriteString(payload.Delta.Reasoning.Text)
			if payload.Delta.Reasoning.Signature != "" || payload.Delta.Reasoning.Text == "" {
				if err := stream.stateBudget.Retain(len(payload.Delta.Reasoning.Signature)); err != nil {
					return fmt.Errorf("Bedrock %w", err)
				}
				builder.signature.WriteString(payload.Delta.Reasoning.Signature)
			}
			if payload.Delta.Reasoning.Text != "" {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: indexID(payload.ContentBlockIndex), Delta: payload.Delta.Reasoning.Text})
			}
		}
		if builder != nil && builder.finalized {
			return fmt.Errorf("Bedrock delta for finalized block %d", payload.ContentBlockIndex)
		}
		if payload.Delta.ToolUse != nil {
			if builder == nil {
				return fmt.Errorf("Bedrock tool delta for unknown block %d", payload.ContentBlockIndex)
			}
			if err := stream.toolBudget.AppendArguments(&builder.input, payload.Delta.ToolUse.Input); err != nil {
				return fmt.Errorf("Bedrock %w", err)
			}
			if !builder.suppressInput {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: builder.id, Delta: payload.Delta.ToolUse.Input})
			}
		}
	case "contentBlockStop":
		var payload struct {
			ContentBlockIndex int `json:"contentBlockIndex"`
		}
		if err := json.Unmarshal(event.Payload, &payload); err != nil {
			return err
		}
		builder := stream.blocks[payload.ContentBlockIndex]
		if builder == nil {
			return nil
		}
		if builder.finalized {
			return nil
		}
		switch builder.kind {
		case "text":
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: indexID(payload.ContentBlockIndex)})
		case "reasoning":
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: indexID(payload.ContentBlockIndex)})
		case "toolUse":
			arguments := json.RawMessage(builder.input.String())
			if len(arguments) == 0 {
				arguments = json.RawMessage(`{}`)
			}
			if !json.Valid(arguments) {
				return fmt.Errorf("Bedrock tool call %q has invalid input", builder.id)
			}
			delete(stream.activeToolIDs, builder.id)
			accepted, err := stream.toolCalls.Accept(builder.id, builder.name, arguments)
			if err != nil {
				return fmt.Errorf("Bedrock %w", err)
			}
			if !accepted {
				break
			}
			call := &llmux.ToolCall{ID: builder.id, Name: builder.name, Arguments: arguments}
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: builder.id}, llmux.Part{Kind: llmux.PartToolCall, ID: builder.id, ToolCall: call})
		}
		builder.finalized = true
	case "messageStop":
		var payload struct {
			StopReason string `json:"stopReason"`
		}
		if err := json.Unmarshal(event.Payload, &payload); err != nil {
			return err
		}
		stream.rawFinish = payload.StopReason
		stream.finish = finishReason(payload.StopReason)
		stream.stopped = true
	case "metadata":
		var payload struct {
			Usage wireUsage `json:"usage"`
		}
		if err := json.Unmarshal(event.Payload, &payload); err != nil {
			return err
		}
		stream.usage = usageResult(payload.Usage)
		if stream.stopped {
			stream.finishPart()
			stream.terminal = true
			_ = stream.Close()
		}
	}
	return nil
}

func (stream *converseStream) finishPart() {
	state := make([]any, 0, len(stream.blocks))
	for index := 0; index < len(stream.blocks); index++ {
		builder := stream.blocks[index]
		if builder == nil {
			continue
		}
		switch builder.kind {
		case "text":
			state = append(state, map[string]any{"text": builder.text.String()})
		case "reasoning":
			state = append(state, map[string]any{"reasoningContent": map[string]any{"reasoningText": map[string]any{"text": builder.text.String(), "signature": builder.signature.String()}}})
		case "toolUse":
			var input any
			_ = json.Unmarshal([]byte(builder.input.String()), &input)
			state = append(state, map[string]any{"toolUse": map[string]any{"toolUseId": builder.id, "name": builder.name, "input": input}})
		}
	}
	stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFinish, FinishReason: stream.finish, RawFinishReason: stream.rawFinish, Usage: stream.usage, ProviderState: mustMarshal(state), Warnings: stream.warnings})
}

func (stream *converseStream) pop() (llmux.Part, bool) {
	if len(stream.pending) == 0 {
		return llmux.Part{}, false
	}
	part := stream.pending[0]
	stream.pending = stream.pending[1:]
	return part, true
}

func (stream *converseStream) popOrEOF() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	return llmux.Part{}, io.EOF
}

func (stream *converseStream) fail(err error) (llmux.Part, error) {
	contextErr := stream.ctx.Err()
	stream.terminal = true
	stream.pending = nil
	_ = stream.Close()
	if contextErr != nil {
		return llmux.Part{}, contextErr
	}
	var providerError *llmux.ProviderError
	if !errors.As(err, &providerError) {
		providerError = &llmux.ProviderError{Provider: "amazon-bedrock", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
}
func indexID(index int) string { return fmt.Sprint(index) }
