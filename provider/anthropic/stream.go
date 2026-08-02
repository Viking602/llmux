package anthropic

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
	kind      string
	id        string
	name      string
	text      strings.Builder
	signature strings.Builder
	input     strings.Builder
}

type messageStream struct {
	ctx        context.Context
	cancel     context.CancelFunc
	body       io.ReadCloser
	reader     *internalstream.SSEReader
	provider   string
	metadata   llmux.ResponseMetadata
	includeRaw bool
	blocks     map[int]*blockBuilder
	pending    []llmux.Part
	usage      llmux.Usage
	finish     llmux.FinishReason
	rawFinish  string
	terminal   bool
	closeOnce  sync.Once
}

func newMessageStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, provider string, metadata llmux.ResponseMetadata, includeRaw bool) *messageStream {
	return &messageStream{ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), provider: provider, metadata: metadata, includeRaw: includeRaw, blocks: make(map[int]*blockBuilder)}
}

func (stream *messageStream) Close() error {
	var err error
	stream.closeOnce.Do(func() {
		stream.cancel()
		err = stream.body.Close()
	})
	return err
}

func (stream *messageStream) Recv() (llmux.Part, error) {
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
				return stream.fail(errors.New("Anthropic stream ended before message_stop"))
			}
			return stream.fail(err)
		}
		data := strings.TrimSpace(frame.Data)
		if data == "" {
			continue
		}
		var event struct {
			Type         string          `json:"type"`
			Index        int             `json:"index"`
			Message      json.RawMessage `json:"message"`
			ContentBlock json.RawMessage `json:"content_block"`
			Delta        json.RawMessage `json:"delta"`
			Usage        usage           `json:"usage"`
			Error        *struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return stream.fail(fmt.Errorf("invalid Anthropic stream JSON: %w", err))
		}
		if event.Type == "" {
			event.Type = frame.Name
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
		}
		if event.Type == "error" {
			message, code := "Anthropic stream failed", ""
			if event.Error != nil {
				message, code = event.Error.Message, event.Error.Type
			}
			return stream.fail(&llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Code: code, Message: message})
		}
		if err := stream.mapEvent(event.Type, event.Index, event.Message, event.ContentBlock, event.Delta, event.Usage); err != nil {
			return stream.fail(err)
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *messageStream) mapEvent(eventType string, index int, messageRaw, blockRaw, deltaRaw json.RawMessage, eventUsage usage) error {
	switch eventType {
	case "message_start":
		var message response
		if err := json.Unmarshal(messageRaw, &message); err != nil {
			return errors.New("invalid Anthropic message_start")
		}
		stream.metadata.ID = message.ID
		stream.metadata.ModelID = first(message.Model, stream.metadata.ModelID)
		stream.usage = usageResult(message.Usage)
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartResponseMetadata, Response: stream.metadata})
	case "content_block_start":
		var block struct {
			Type     string          `json:"type"`
			ID       string          `json:"id"`
			Name     string          `json:"name"`
			Text     string          `json:"text"`
			Thinking string          `json:"thinking"`
			Input    json.RawMessage `json:"input"`
			Data     string          `json:"data"`
		}
		if err := json.Unmarshal(blockRaw, &block); err != nil {
			return errors.New("invalid Anthropic content_block_start")
		}
		builder := &blockBuilder{kind: block.Type, id: block.ID, name: block.Name}
		stream.blocks[index] = builder
		switch block.Type {
		case "text":
			builder.text.WriteString(block.Text)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: fmt.Sprint(index)})
			if block.Text != "" {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: fmt.Sprint(index), Delta: block.Text})
			}
		case "thinking":
			builder.text.WriteString(block.Thinking)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningStart, ID: fmt.Sprint(index)})
			if block.Thinking != "" {
				stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: fmt.Sprint(index), Delta: block.Thinking})
			}
		case "tool_use", "server_tool_use":
			if len(block.Input) > 0 && string(block.Input) != "{}" {
				builder.input.Write(block.Input)
			}
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: block.ID, ToolName: block.Name})
		case "redacted_thinking":
			builder.text.WriteString(block.Data)
		}
	case "content_block_delta":
		builder := stream.blocks[index]
		if builder == nil {
			return fmt.Errorf("Anthropic delta for unknown block %d", index)
		}
		var delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			Signature   string `json:"signature"`
			PartialJSON string `json:"partial_json"`
		}
		if err := json.Unmarshal(deltaRaw, &delta); err != nil {
			return errors.New("invalid Anthropic content block delta")
		}
		switch delta.Type {
		case "text_delta":
			builder.text.WriteString(delta.Text)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: fmt.Sprint(index), Delta: delta.Text})
		case "thinking_delta":
			builder.text.WriteString(delta.Thinking)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: fmt.Sprint(index), Delta: delta.Thinking})
		case "signature_delta":
			builder.signature.WriteString(delta.Signature)
		case "input_json_delta":
			builder.input.WriteString(delta.PartialJSON)
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputDelta, ID: builder.id, Delta: delta.PartialJSON})
		}
	case "content_block_stop":
		builder := stream.blocks[index]
		if builder == nil {
			return fmt.Errorf("Anthropic stop for unknown block %d", index)
		}
		switch builder.kind {
		case "text":
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: fmt.Sprint(index)})
		case "thinking":
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: fmt.Sprint(index)})
		case "tool_use", "server_tool_use":
			arguments := json.RawMessage(builder.input.String())
			if len(arguments) == 0 {
				arguments = json.RawMessage(`{}`)
			}
			if !json.Valid(arguments) {
				return fmt.Errorf("Anthropic tool call %q has invalid input", builder.id)
			}
			call := &llmux.ToolCall{ID: builder.id, Name: builder.name, Arguments: arguments}
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputEnd, ID: builder.id}, llmux.Part{Kind: llmux.PartToolCall, ID: builder.id, ToolCall: call})
		}
	case "message_delta":
		var delta struct {
			StopReason string `json:"stop_reason"`
		}
		if err := json.Unmarshal(deltaRaw, &delta); err != nil {
			return errors.New("invalid Anthropic message delta")
		}
		if delta.StopReason != "" {
			stream.rawFinish = delta.StopReason
			stream.finish = finishReason(delta.StopReason)
		}
		if eventUsage.OutputTokens > 0 {
			stream.usage.OutputTokens = eventUsage.OutputTokens
			stream.usage.ReasoningTokens = eventUsage.OutputTokensDetails.ThinkingTokens
			stream.usage.TotalTokens = stream.usage.InputTokens + stream.usage.OutputTokens
		}
	case "message_stop":
		if stream.finish == "" {
			stream.finish = llmux.FinishUnknown
		}
		state, err := stream.providerState()
		if err != nil {
			return err
		}
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFinish, FinishReason: stream.finish, RawFinishReason: stream.rawFinish, Usage: stream.usage, ProviderState: state})
		stream.terminal = true
		_ = stream.Close()
	case "ping":
	}
	return nil
}

func (stream *messageStream) providerState() (json.RawMessage, error) {
	blocks := make([]any, 0, len(stream.blocks))
	for index := 0; index < len(stream.blocks); index++ {
		builder := stream.blocks[index]
		if builder == nil {
			continue
		}
		switch builder.kind {
		case "text":
			blocks = append(blocks, map[string]any{"type": "text", "text": builder.text.String()})
		case "thinking":
			blocks = append(blocks, map[string]any{"type": "thinking", "thinking": builder.text.String(), "signature": builder.signature.String()})
		case "redacted_thinking":
			blocks = append(blocks, map[string]any{"type": "redacted_thinking", "data": builder.text.String()})
		case "tool_use", "server_tool_use":
			input := json.RawMessage(builder.input.String())
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			if !json.Valid(input) {
				return nil, fmt.Errorf("Anthropic tool call %q has invalid input", builder.id)
			}
			var value any
			_ = json.Unmarshal(input, &value)
			blocks = append(blocks, map[string]any{"type": builder.kind, "id": builder.id, "name": builder.name, "input": value})
		}
	}
	return json.Marshal(blocks)
}

func (stream *messageStream) pop() (llmux.Part, bool) {
	if len(stream.pending) == 0 {
		return llmux.Part{}, false
	}
	part := stream.pending[0]
	stream.pending = stream.pending[1:]
	return part, true
}

func (stream *messageStream) fail(err error) (llmux.Part, error) {
	stream.terminal = true
	_ = stream.Close()
	if contextErr := stream.ctx.Err(); contextErr != nil {
		return llmux.Part{}, contextErr
	}
	var providerError *llmux.ProviderError
	if errors.As(err, &providerError) {
		return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
	}
	providerError = &llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
}
