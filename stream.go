package llmux

import (
	"errors"
	"fmt"
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

// TextPhase distinguishes model-authored commentary from terminal answer text.
type TextPhase string

const (
	// TextPhaseUnspecified is the legacy provider path and is interpreted as a
	// final answer. Providers that expose intermediate prose use Commentary.
	TextPhaseUnspecified TextPhase = ""
	TextPhaseCommentary  TextPhase = "commentary"
	TextPhaseFinalAnswer TextPhase = "final_answer"
)

type Part struct {
	Kind            PartKind         `json:"kind"`
	ID              string           `json:"id,omitempty"`
	ToolName        string           `json:"toolName,omitempty"`
	Delta           string           `json:"delta,omitempty"`
	Text            string           `json:"text,omitempty"`
	TextPhase       TextPhase        `json:"textPhase,omitempty"`
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

const (
	MaxStreamParts     = 65_536
	MaxStreamBytes     = 64 << 20
	MaxStreamToolCalls = 1024
)

var (
	ErrMissingStreamTerminal  = errors.New("llmux: provider stream ended without a terminal part")
	ErrStreamAfterTerminal    = errors.New("llmux: provider stream emitted data after terminal part")
	ErrMultipleStreamTerminal = errors.New("llmux: provider stream emitted multiple terminal parts")
	ErrStreamLimit            = errors.New("llmux: provider stream exceeds safe limits")
	ErrInvalidStreamToolCall  = errors.New("llmux: provider stream emitted an invalid tool call")
)

// ConformStream enforces provider-neutral terminal, size, and tool identity
// invariants for callers that consume a stream incrementally.
func ConformStream(stream Stream) Stream {
	if stream == nil {
		return nil
	}
	if _, already := stream.(*conformingStream); already {
		return stream
	}
	return &conformingStream{source: stream, toolIDs: make(map[string]struct{})}
}

type conformingStream struct {
	source    Stream
	terminal  bool
	failure   error
	parts     int
	bytes     int
	toolCalls int
	toolIDs   map[string]struct{}
}

func (stream *conformingStream) Recv() (Part, error) {
	if stream.failure != nil {
		return Part{}, stream.failure
	}
	part, err := stream.source.Recv()
	if errors.Is(err, io.EOF) {
		if !stream.terminal {
			return stream.fail(ErrMissingStreamTerminal)
		}
		return Part{}, io.EOF
	}
	if err != nil {
		return stream.fail(err)
	}
	if stream.terminal {
		if part.Kind == PartFinish || part.Kind == PartError {
			return stream.fail(ErrMultipleStreamTerminal)
		}
		return stream.fail(ErrStreamAfterTerminal)
	}
	stream.parts++
	size := streamPartSize(part)
	if stream.parts > MaxStreamParts || size > MaxStreamBytes-stream.bytes {
		return stream.fail(fmt.Errorf("%w: parts=%d bytes>%d", ErrStreamLimit, stream.parts, MaxStreamBytes))
	}
	stream.bytes += size
	if part.Kind == PartToolCall {
		if part.ToolCall == nil {
			return stream.fail(ErrInvalidStreamToolCall)
		}
		stream.toolCalls++
		if stream.toolCalls > MaxStreamToolCalls {
			return stream.fail(fmt.Errorf("%w: more than %d tool calls", ErrStreamLimit, MaxStreamToolCalls))
		}
		if strings.TrimSpace(part.ToolCall.ID) == "" {
			call := *part.ToolCall
			generated, err := stream.generatedToolID()
			if err != nil {
				return stream.fail(fmt.Errorf("generate tool call id: %w", err))
			}
			call.ID = generated
			part.ToolCall = &call
			if part.ID == "" {
				part.ID = call.ID
			}
		}
		if _, duplicate := stream.toolIDs[part.ToolCall.ID]; duplicate {
			return stream.fail(fmt.Errorf("%w: duplicate id %q", ErrInvalidStreamToolCall, part.ToolCall.ID))
		}
		stream.toolIDs[part.ToolCall.ID] = struct{}{}
	}
	if part.Kind == PartFinish || part.Kind == PartError {
		stream.terminal = true
	}
	return part, nil
}

func (stream *conformingStream) generatedToolID() (string, error) {
	for {
		candidate, err := NewGeneratedToolCallID("llmux")
		if err != nil {
			return "", err
		}
		if _, exists := stream.toolIDs[candidate]; !exists {
			return candidate, nil
		}
	}
}

func (stream *conformingStream) fail(err error) (Part, error) {
	stream.failure = err
	_ = stream.source.Close()
	return Part{}, err
}

func (stream *conformingStream) Close() error {
	return stream.source.Close()
}

func streamPartSize(part Part) int {
	size := len(part.ID) + len(part.ToolName) + len(part.Delta) + len(part.Text) +
		len(part.RawFinishReason) + len(part.ProviderState) + len(part.Raw) +
		len(part.Usage.Raw) + len(part.Response.ID) + len(part.Response.ModelID)
	for key, value := range part.Response.Headers {
		size += len(key) + len(value)
	}
	if part.ToolCall != nil {
		size += len(part.ToolCall.ID) + len(part.ToolCall.Name) + len(part.ToolCall.Arguments)
	}
	if part.ToolResult != nil {
		size += len(part.ToolResult.ToolCallID) + len(part.ToolResult.Name) +
			len(part.ToolResult.Content) + len(part.ToolResult.Structured)
	}
	if part.Content != nil {
		size += len(part.Content.Text) + len(part.Content.Data) + len(part.Content.URL) +
			len(part.Content.MediaType) + len(part.Content.Filename) + len(part.Content.ProviderData)
		if part.Content.ToolCall != nil {
			size += len(part.Content.ToolCall.ID) + len(part.Content.ToolCall.Name) +
				len(part.Content.ToolCall.Arguments)
		}
		if part.Content.ToolResult != nil {
			size += len(part.Content.ToolResult.ToolCallID) + len(part.Content.ToolResult.Name) +
				len(part.Content.ToolResult.Content) + len(part.Content.ToolResult.Structured)
		}
		if part.Content.Source != nil {
			size += len(part.Content.Source.ID) + len(part.Content.Source.URL) +
				len(part.Content.Source.Title) + len(part.Content.Source.MediaType) +
				len(part.Content.Source.Filename)
		}
	}
	for _, warning := range part.Warnings {
		size += len(warning)
	}
	return size
}

// Collect drains a stream into a non-streaming result. It is primarily useful
// for providers whose native API only streams and for test/conformance code.
func Collect(stream Stream) (result Result, err error) {
	if stream == nil {
		return Result{}, errors.New("llmux: nil stream")
	}
	stream = ConformStream(stream)
	defer func() { err = errors.Join(err, stream.Close()) }()

	var text, reasoning strings.Builder
	content := streamContentCollector{}
	sawTerminal := false
	for {
		part, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			return Result{}, recvErr
		}
		if sawTerminal && part.Kind != PartRaw && part.Kind != PartFinish {
			return Result{}, errors.New("llmux: provider stream emitted content after finish")
		}
		switch part.Kind {
		case PartTextDelta:
			text.WriteString(part.Delta)
			content.append(textContentKind(part.TextPhase), part.Delta)
		case PartReasoningDelta:
			reasoning.WriteString(part.Delta)
			content.append(ContentReasoning, part.Delta)
		case PartToolCall:
			if part.ToolCall != nil {
				result.ToolCalls = append(result.ToolCalls, *part.ToolCall)
			}
		case PartFile, PartSource:
			if part.Content != nil {
				content.appendPart(*part.Content)
			}
		case PartResponseMetadata:
			result.Response = part.Response
		case PartFinish:
			if sawTerminal {
				return Result{}, errors.New("llmux: provider stream emitted multiple finish parts")
			}
			sawTerminal = true
			result.Usage = NormalizeUsage(part.Usage)
			result.FinishReason = part.FinishReason
			result.RawFinishReason = part.RawFinishReason
			result.ProviderState = append(result.ProviderState[:0], part.ProviderState...)
			result.Warnings = append(result.Warnings, part.Warnings...)
		case PartError:
			if part.Err != nil {
				return Result{}, part.Err
			}
			return Result{}, errors.New("llmux: provider stream error")
		default:
		}
	}
	if !sawTerminal {
		return Result{}, errors.New("llmux: provider stream ended without a finish part")
	}
	result.Text = text.String()
	result.Reasoning = reasoning.String()
	result.Content = content.content()
	if result.FinishReason == "" {
		result.FinishReason = FinishUnknown
	}
	return result, nil
}

func textContentKind(phase TextPhase) ContentKind {
	switch phase {
	case TextPhaseCommentary:
		return ContentCommentary
	case TextPhaseFinalAnswer, TextPhaseUnspecified:
		return ContentFinalAnswer
	default:
		return ContentFinalAnswer
	}
}

type streamContentCollector struct {
	parts []*streamContentPart
}

type streamContentPart struct {
	content ContentPart
	text    strings.Builder
}

func (collector *streamContentCollector) append(kind ContentKind, delta string) {
	if delta == "" {
		return
	}
	if len(collector.parts) > 0 && collector.parts[len(collector.parts)-1].content.Kind == kind {
		collector.parts[len(collector.parts)-1].text.WriteString(delta)
		return
	}
	part := &streamContentPart{content: ContentPart{Kind: kind}}
	part.text.WriteString(delta)
	collector.parts = append(collector.parts, part)
}

func (collector *streamContentCollector) appendPart(part ContentPart) {
	part.Data = append([]byte(nil), part.Data...)
	part.ProviderData = append([]byte(nil), part.ProviderData...)
	if part.ToolCall != nil {
		call := *part.ToolCall
		call.Arguments = append([]byte(nil), call.Arguments...)
		part.ToolCall = &call
	}
	if part.ToolResult != nil {
		result := *part.ToolResult
		result.Structured = append([]byte(nil), result.Structured...)
		part.ToolResult = &result
	}
	if part.Source != nil {
		source := *part.Source
		part.Source = &source
	}
	collector.parts = append(collector.parts, &streamContentPart{content: part})
}

func (collector *streamContentCollector) content() []ContentPart {
	parts := make([]ContentPart, len(collector.parts))
	for index, part := range collector.parts {
		parts[index] = part.content
		if part.text.Len() > 0 {
			parts[index].Text = part.text.String()
		}
	}
	return parts
}
