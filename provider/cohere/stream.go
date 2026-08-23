package cohere

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

type toolBuilder struct {
	id            string
	name          string
	arguments     strings.Builder
	suppressInput bool
}
type chatStream struct {
	ctx              context.Context
	cancel           context.CancelFunc
	body             io.ReadCloser
	reader           *internalstream.SSEReader
	includeRaw       bool
	pending          []llmux.Part
	tools            map[int]*toolBuilder
	startedIndexes   map[int]bool
	completedIndexes map[int]bool
	activeToolIDs    map[string]int
	usage            llmux.Usage
	finish           llmux.FinishReason
	rawFinish        string
	reasoningIndexes map[int]bool
	completed        bool
	terminal         bool
	closeOnce        sync.Once
	toolCalls        internalstream.ToolCallTracker
	toolBudget       internalstream.ToolCallBudget
}

func newChatStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, includeRaw bool) *chatStream {
	return &chatStream{
		ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), includeRaw: includeRaw,
		tools: make(map[int]*toolBuilder), startedIndexes: make(map[int]bool),
		completedIndexes: make(map[int]bool), activeToolIDs: make(map[string]int),
		reasoningIndexes: make(map[int]bool),
	}
}

func (stream *chatStream) Close() error {
	var err error
	stream.closeOnce.Do(func() {
		stream.cancel()
		err = stream.body.Close()
	})
	return err
}

func (stream *chatStream) Recv() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	if stream.terminal {
		return llmux.Part{}, io.EOF
	}
	for {
		frame, err := stream.reader.Next()
		if err != nil {
			if errors.Is(err, io.EOF) && stream.completed && stream.ctx.Err() == nil {
				stream.terminal = true
				_ = stream.Close()
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFinish, FinishReason: stream.finish, RawFinishReason: stream.rawFinish, Usage: stream.usage})
				return stream.popOrEOF()
			}
			return stream.fail(err)
		}
		data := strings.TrimSpace(frame.Data)
		if data == "" {
			continue
		}
		var event struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			ID    string `json:"id"`
			Delta struct {
				Message struct {
					Content   map[string]any `json:"content"`
					ToolCalls map[string]any `json:"tool_calls"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
				Usage        *struct {
					Tokens tokenPair `json:"tokens"`
				} `json:"usage"`
			} `json:"delta"`
		}
		if err := internalstream.ValidateJSONComplexity([]byte(data)); err != nil {
			return stream.fail(err)
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return stream.fail(err)
		}
		if event.Type == "" {
			event.Type = frame.Name
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
		}
		if stream.completed {
			return stream.fail(fmt.Errorf("Cohere event %q arrived after message-end", event.Type))
		}
		switch event.Type {
		case "message-start":
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartResponseMetadata, Response: llmux.ResponseMetadata{ID: event.ID}})
		case "content-start":
			if event.Delta.Message.Content["type"] == "thinking" {
				stream.reasoningIndexes[event.Index] = true
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningStart, ID: indexID(event.Index)})
			} else {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: indexID(event.Index)})
			}
		case "content-delta":
			if thinking, ok := event.Delta.Message.Content["thinking"].(string); ok {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: indexID(event.Index), Delta: thinking})
			} else if text, ok := event.Delta.Message.Content["text"].(string); ok {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: indexID(event.Index), Delta: text})
			}
		case "content-end":
			if stream.reasoningIndexes[event.Index] {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: indexID(event.Index)})
			} else {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: indexID(event.Index)})
			}
		case "tool-call-start":
			if stream.startedIndexes[event.Index] {
				return stream.fail(fmt.Errorf("Cohere tool index %d was started more than once", event.Index))
			}
			if err := stream.toolBudget.Begin(); err != nil {
				return stream.fail(fmt.Errorf("Cohere %w", err))
			}
			function, _ := event.Delta.Message.ToolCalls["function"].(map[string]any)
			builder := &toolBuilder{id: stringValue(event.Delta.Message.ToolCalls["id"]), name: stringValue(function["name"])}
			if err := stream.toolBudget.AddAlias(builder.id); err != nil {
				return stream.fail(fmt.Errorf("Cohere %w", err))
			}
			if err := stream.toolBudget.AddMetadata(builder.name); err != nil {
				return stream.fail(fmt.Errorf("Cohere %w", err))
			}
			if builder.id != "" {
				if activeIndex, exists := stream.activeToolIDs[builder.id]; exists {
					return stream.fail(fmt.Errorf("Cohere tool call ID %q is already active at index %d", builder.id, activeIndex))
				}
				stream.activeToolIDs[builder.id] = event.Index
			}
			builder.suppressInput = stream.toolCalls.Seen(builder.id)
			initial := stringValue(function["arguments"])
			if err := stream.toolBudget.AppendArguments(&builder.arguments, initial); err != nil {
				return stream.fail(fmt.Errorf("Cohere %w", err))
			}
			stream.tools[event.Index] = builder
			stream.startedIndexes[event.Index] = true
			if !builder.suppressInput {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: builder.id, ToolName: builder.name})
				if initial != "" {
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: builder.id, Delta: initial})
				}
			}
		case "tool-call-delta":
			builder := stream.tools[event.Index]
			if builder == nil {
				return stream.fail(fmt.Errorf("Cohere tool delta for unknown index %d", event.Index))
			}
			function, _ := event.Delta.Message.ToolCalls["function"].(map[string]any)
			delta := stringValue(function["arguments"])
			if err := stream.toolBudget.AppendArguments(&builder.arguments, delta); err != nil {
				return stream.fail(fmt.Errorf("Cohere %w", err))
			}
			if !builder.suppressInput {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: builder.id, Delta: delta})
			}
		case "tool-call-end":
			if stream.completedIndexes[event.Index] {
				break
			}
			builder := stream.tools[event.Index]
			if builder == nil {
				return stream.fail(fmt.Errorf("Cohere tool end for unknown index %d", event.Index))
			}
			arguments := json.RawMessage(builder.arguments.String())
			if len(arguments) == 0 || string(arguments) == "null" {
				arguments = json.RawMessage(`{}`)
			}
			if !json.Valid(arguments) {
				return stream.fail(errors.New("Cohere streamed invalid tool arguments"))
			}
			accepted, err := stream.toolCalls.Accept(builder.id, builder.name, arguments)
			if err != nil {
				return stream.fail(fmt.Errorf("Cohere %w", err))
			}
			if accepted {
				call := &llmux.ToolCall{ID: builder.id, Name: builder.name, Arguments: arguments}
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: call.ID}, llmux.Part{Kind: llmux.PartToolCall, ID: call.ID, ToolCall: call})
			}
			delete(stream.activeToolIDs, builder.id)
			delete(stream.tools, event.Index)
			stream.completedIndexes[event.Index] = true
		case "message-end":
			stream.rawFinish = event.Delta.FinishReason
			stream.finish = finishReason(stream.rawFinish)
			if event.Delta.Usage != nil {
				stream.usage = convertUsage(event.Delta.Usage.Tokens)
			}
			stream.completed = true
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *chatStream) pop() (llmux.Part, bool) {
	if len(stream.pending) == 0 {
		return llmux.Part{}, false
	}
	part := stream.pending[0]
	stream.pending = stream.pending[1:]
	return part, true
}

func (stream *chatStream) popOrEOF() (llmux.Part, error) {
	if part, ok := stream.pop(); ok {
		return part, nil
	}
	return llmux.Part{}, io.EOF
}

func (stream *chatStream) fail(err error) (llmux.Part, error) {
	contextErr := stream.ctx.Err()
	stream.terminal = true
	stream.pending = nil
	_ = stream.Close()
	if contextErr != nil {
		return llmux.Part{}, contextErr
	}
	providerError := &llmux.ProviderError{Provider: "cohere", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
}

func indexID(index int) string { return fmt.Sprint(index) }

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}
