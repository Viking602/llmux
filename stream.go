package llmux

import (
	"errors"
	"io"
	"strings"
)

type PartKind string

const (
	PartStreamStart      PartKind = "stream_start"
	PartResponseMetadata PartKind = "response_metadata"
	PartTextStart        PartKind = "text_start"
	PartTextDelta        PartKind = "text_delta"
	PartTextEnd          PartKind = "text_end"
	PartReasoningStart   PartKind = "reasoning_start"
	PartReasoningDelta   PartKind = "reasoning_delta"
	PartReasoningEnd     PartKind = "reasoning_end"
	PartToolInputStart   PartKind = "tool_input_start"
	PartToolInputDelta   PartKind = "tool_input_delta"
	PartToolInputEnd     PartKind = "tool_input_end"
	PartToolCall         PartKind = "tool_call"
	PartToolResult       PartKind = "tool_result"
	PartFile             PartKind = "file"
	PartSource           PartKind = "source"
	PartFinish           PartKind = "finish"
	PartError            PartKind = "error"
	PartRaw              PartKind = "raw"
)

type Part struct {
	Kind            PartKind         `json:"kind"`
	ID              string           `json:"id,omitempty"`
	ToolName        string           `json:"toolName,omitempty"`
	Delta           string           `json:"delta,omitempty"`
	Text            string           `json:"text,omitempty"`
	ToolCall        *ToolCall        `json:"toolCall,omitempty"`
	ToolResult      *ToolResult      `json:"toolResult,omitempty"`
	Content         *ContentPart     `json:"content,omitempty"`
	Usage           Usage            `json:"usage,omitempty"`
	FinishReason    FinishReason     `json:"finishReason,omitempty"`
	RawFinishReason string           `json:"rawFinishReason,omitempty"`
	Response        ResponseMetadata `json:"response,omitempty"`
	ProviderState   []byte           `json:"providerState,omitempty"`
	Warnings        []string         `json:"warnings,omitempty"`
	Raw             []byte           `json:"raw,omitempty"`
	Err             error            `json:"-"`
}

type Stream interface {
	Recv() (Part, error)
	Close() error
}

// Collect drains a stream into a non-streaming result. It is primarily useful
// for providers whose native API only streams and for test/conformance code.
func Collect(stream Stream) (result Result, err error) {
	if stream == nil {
		return Result{}, errors.New("llmux: nil stream")
	}
	defer func() { err = errors.Join(err, stream.Close()) }()

	var text, reasoning strings.Builder
	for {
		part, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			return Result{}, recvErr
		}
		switch part.Kind {
		case PartTextDelta:
			text.WriteString(part.Delta)
		case PartReasoningDelta:
			reasoning.WriteString(part.Delta)
		case PartToolCall:
			if part.ToolCall != nil {
				result.ToolCalls = append(result.ToolCalls, *part.ToolCall)
			}
		case PartFile, PartSource:
			if part.Content != nil {
				result.Content = append(result.Content, *part.Content)
			}
		case PartResponseMetadata:
			result.Response = part.Response
		case PartFinish:
			result.Usage = part.Usage
			result.FinishReason = part.FinishReason
			result.RawFinishReason = part.RawFinishReason
			result.ProviderState = append(result.ProviderState[:0], part.ProviderState...)
			result.Warnings = append(result.Warnings, part.Warnings...)
		case PartError:
			if part.Err != nil {
				return Result{}, part.Err
			}
			return Result{}, errors.New("llmux: provider stream error")
		}
	}
	result.Text = text.String()
	result.Reasoning = reasoning.String()
	if result.Text != "" {
		result.Content = append([]ContentPart{{Kind: ContentText, Text: result.Text}}, result.Content...)
	}
	if result.Reasoning != "" {
		result.Content = append([]ContentPart{{Kind: ContentReasoning, Text: result.Reasoning}}, result.Content...)
	}
	if result.FinishReason == "" {
		result.FinishReason = FinishUnknown
	}
	return result, nil
}
