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
	id        string
	name      string
	arguments strings.Builder
}

type chatStream struct {
	ctx              context.Context
	cancel           context.CancelFunc
	body             io.ReadCloser
	reader           *internalstream.SSEReader
	includeRaw       bool
	pending          []llmux.Part
	tool             *toolBuilder
	usage            llmux.Usage
	finish           llmux.FinishReason
	rawFinish        string
	reasoningIndexes map[int]bool
	completed        bool
	terminal         bool
	closeOnce        sync.Once
}

func newChatStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, includeRaw bool) *chatStream {
	return &chatStream{ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), includeRaw: includeRaw, reasoningIndexes: make(map[int]bool)}
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
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return stream.fail(err)
		}
		if event.Type == "" {
			event.Type = frame.Name
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
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
			function, _ := event.Delta.Message.ToolCalls["function"].(map[string]any)
			stream.tool = &toolBuilder{id: stringValue(event.Delta.Message.ToolCalls["id"]), name: stringValue(function["name"])}
			initial := stringValue(function["arguments"])
			stream.tool.arguments.WriteString(initial)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: stream.tool.id, ToolName: stream.tool.name})
			if initial != "" {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: stream.tool.id, Delta: initial})
			}
		case "tool-call-delta":
			if stream.tool != nil {
				function, _ := event.Delta.Message.ToolCalls["function"].(map[string]any)
				delta := stringValue(function["arguments"])
				stream.tool.arguments.WriteString(delta)
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: stream.tool.id, Delta: delta})
			}
		case "tool-call-end":
			if stream.tool != nil {
				arguments := json.RawMessage(stream.tool.arguments.String())
				if len(arguments) == 0 || string(arguments) == "null" {
					arguments = json.RawMessage(`{}`)
				}
				if !json.Valid(arguments) {
					return stream.fail(errors.New("Cohere streamed invalid tool arguments"))
				}
				call := &llmux.ToolCall{ID: stream.tool.id, Name: stream.tool.name, Arguments: arguments}
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: call.ID}, llmux.Part{Kind: llmux.PartToolCall, ID: call.ID, ToolCall: call})
				stream.tool = nil
			}
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
	stream.terminal = true
	_ = stream.Close()
	if contextErr := stream.ctx.Err(); contextErr != nil {
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
