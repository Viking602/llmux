package google

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"

	"github.com/Viking602/llmux"
	internalstream "github.com/Viking602/llmux/internal/stream"
)

type geminiStream struct {
	ctx              context.Context
	cancel           context.CancelFunc
	body             io.ReadCloser
	reader           *internalstream.SSEReader
	provider         string
	metadata         llmux.ResponseMetadata
	warnings         []string
	includeRaw       bool
	pending          []llmux.Part
	usage            llmux.Usage
	finish           llmux.FinishReason
	rawFinish        string
	providerParts    []json.RawMessage
	textStarted      bool
	reasoningStarted bool
	completed        bool
	toolUse          bool
	terminal         bool
	closeOnce        sync.Once
}

func newGeminiStream(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser, provider string, metadata llmux.ResponseMetadata, warnings []string, includeRaw bool) *geminiStream {
	return &geminiStream{ctx: ctx, cancel: cancel, body: body, reader: internalstream.NewSSEReader(body, 0), provider: provider, metadata: metadata, warnings: warnings, includeRaw: includeRaw}
}

func (stream *geminiStream) Close() error {
	var err error
	stream.closeOnce.Do(func() {
		stream.cancel()
		err = stream.body.Close()
	})
	return err
}

func (stream *geminiStream) Recv() (llmux.Part, error) {
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
				stream.finishParts()
				stream.terminal = true
				_ = stream.Close()
				if part, ok := stream.pop(); ok {
					return part, nil
				}
				return llmux.Part{}, io.EOF
			}
			return stream.fail(err)
		}
		data := strings.TrimSpace(frame.Data)
		if data == "" {
			continue
		}
		var chunk wireResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return stream.fail(err)
		}
		if stream.includeRaw {
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartRaw, Raw: []byte(data)})
		}
		if stream.metadata.ID == "" && chunk.ResponseID != "" {
			stream.metadata.ID = chunk.ResponseID
			if chunk.ModelVersion != "" {
				stream.metadata.ModelID = chunk.ModelVersion
			}
			stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartResponseMetadata, Response: stream.metadata})
		}
		if chunk.UsageMetadata.TotalTokenCount > 0 || chunk.UsageMetadata.PromptTokenCount > 0 || chunk.UsageMetadata.CandidatesTokenCount > 0 {
			stream.usage = convertUsage(chunk.UsageMetadata)
		}
		if len(chunk.Candidates) > 0 {
			candidate := chunk.Candidates[0]
			for _, raw := range candidate.Content.Parts {
				stream.providerParts = append(stream.providerParts, append(json.RawMessage(nil), raw...))
				var part struct {
					Text         *string `json:"text"`
					Thought      bool    `json:"thought"`
					FunctionCall *struct {
						ID   string          `json:"id"`
						Name string          `json:"name"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
					InlineData *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData"`
				}
				if err := json.Unmarshal(raw, &part); err != nil {
					return stream.fail(err)
				}
				if part.Text != nil {
					if part.Thought {
						if !stream.reasoningStarted {
							stream.reasoningStarted = true
							stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningStart, ID: "reasoning-0"})
						}
						stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningDelta, ID: "reasoning-0", Delta: *part.Text})
					} else {
						if !stream.textStarted {
							stream.textStarted = true
							stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextStart, ID: "text-0"})
						}
						stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextDelta, ID: "text-0", Delta: *part.Text})
					}
				}
				if part.FunctionCall != nil {
					stream.toolUse = true
					arguments := part.FunctionCall.Args
					if len(arguments) == 0 {
						arguments = json.RawMessage(`{}`)
					}
					if !json.Valid(arguments) {
						return stream.fail(errors.New("Google streamed invalid function arguments"))
					}
					call := &llmux.ToolCall{ID: part.FunctionCall.ID, Name: part.FunctionCall.Name, Arguments: arguments}
					stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartToolInputStart, ID: call.ID, ToolName: call.Name}, llmux.Part{Kind: llmux.PartToolInputDelta, ID: call.ID, Delta: string(arguments)}, llmux.Part{Kind: llmux.PartToolInputEnd, ID: call.ID}, llmux.Part{Kind: llmux.PartToolCall, ID: call.ID, ToolCall: call})
				}
				if part.InlineData != nil {
					result := llmux.Result{}
					if err := appendParts(&result, []json.RawMessage{raw}); err != nil {
						return stream.fail(err)
					}
					for index := range result.Content {
						content := result.Content[index]
						stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFile, Content: &content})
					}
				}
			}
			if candidate.FinishReason != "" {
				stream.finish, stream.rawFinish = finishReason(candidate.FinishReason, stream.toolUse)
				stream.completed = true
			}
		}
		if part, ok := stream.pop(); ok {
			return part, nil
		}
	}
}

func (stream *geminiStream) finishParts() {
	if stream.textStarted {
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartTextEnd, ID: "text-0"})
	}
	if stream.reasoningStarted {
		stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartReasoningEnd, ID: "reasoning-0"})
	}
	stream.pending = append(stream.pending, llmux.Part{Kind: llmux.PartFinish, FinishReason: stream.finish, RawFinishReason: stream.rawFinish, Usage: stream.usage, ProviderState: mustMarshal(stream.providerParts), Warnings: stream.warnings})
}

func (stream *geminiStream) pop() (llmux.Part, bool) {
	if len(stream.pending) == 0 {
		return llmux.Part{}, false
	}
	part := stream.pending[0]
	stream.pending = stream.pending[1:]
	return part, true
}

func (stream *geminiStream) fail(err error) (llmux.Part, error) {
	stream.terminal = true
	_ = stream.Close()
	if contextErr := stream.ctx.Err(); contextErr != nil {
		return llmux.Part{}, contextErr
	}
	providerError := &llmux.ProviderError{Provider: stream.provider, Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	return llmux.Part{Kind: llmux.PartError, Err: providerError}, nil
}
