package llmux

import (
	"errors"
	"io"
	"strings"
	"testing"
)

type sliceStream struct {
	parts  []Part
	closed bool
}

func (stream *sliceStream) Recv() (Part, error) {
	if len(stream.parts) == 0 {
		return Part{}, io.EOF
	}
	part := stream.parts[0]
	stream.parts = stream.parts[1:]
	return part, nil
}

func (stream *sliceStream) Close() error {
	stream.closed = true
	return nil
}

func TestCollect(t *testing.T) {
	stream := &sliceStream{parts: []Part{
		{Kind: PartTextDelta, Delta: "hel"},
		{Kind: PartTextDelta, Delta: "lo"},
		{Kind: PartReasoningDelta, Delta: "think"},
		{Kind: PartFinish, FinishReason: FinishStop, Usage: Usage{TotalTokens: 3}},
	}}
	result, err := Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.Reasoning != "think" || result.FinishReason != FinishStop || result.Usage.TotalTokens != 3 || !stream.closed {
		t.Fatalf("result/closed = %#v/%v", result, stream.closed)
	}
}

func TestCollectPreservesTextPhasesInOrder(t *testing.T) {
	stream := &sliceStream{parts: []Part{
		{Kind: PartTextDelta, Delta: "checking", TextPhase: TextPhaseCommentary},
		{Kind: PartTextDelta, Delta: "done", TextPhase: TextPhaseFinalAnswer},
		{Kind: PartFinish, FinishReason: FinishStop},
	}}
	result, err := Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "checkingdone" || len(result.Content) != 2 ||
		result.Content[0].Kind != ContentCommentary || result.Content[0].Text != "checking" ||
		result.Content[1].Kind != ContentFinalAnswer || result.Content[1].Text != "done" {
		t.Fatalf("phased result = %#v", result)
	}
}

func TestCollectRequiresOneFinalFinish(t *testing.T) {
	for name, parts := range map[string][]Part{
		"missing": {{Kind: PartTextDelta, Delta: "partial"}},
		"duplicate": {
			{Kind: PartFinish, FinishReason: FinishStop},
			{Kind: PartFinish, FinishReason: FinishStop},
		},
		"post-finish": {
			{Kind: PartFinish, FinishReason: FinishStop},
			{Kind: PartTextDelta, Delta: "late"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Collect(&sliceStream{parts: parts})
			if err == nil || errors.Is(err, io.EOF) {
				t.Fatalf("terminal sequence error = %v", err)
			}
		})
	}
}

func TestConformStreamRejectsDuplicateToolCallIdentityAndCloses(t *testing.T) {
	source := &sliceStream{parts: []Part{
		{Kind: PartToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup"}},
		{Kind: PartToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup"}},
		{Kind: PartFinish, FinishReason: FinishToolCalls},
	}}
	stream := ConformStream(source)
	if _, err := stream.Recv(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Recv(); !errors.Is(err, ErrInvalidStreamToolCall) {
		t.Fatalf("duplicate tool call error = %v", err)
	}
	if !source.closed {
		t.Fatal("conformance failure did not close the provider stream")
	}
}

func TestConformStreamSynthesizesIdentityForProtocolAllowedIDLessCall(t *testing.T) {
	stream := ConformStream(&sliceStream{parts: []Part{
		{Kind: PartToolCall, ToolCall: &ToolCall{Name: "lookup"}},
		{Kind: PartFinish, FinishReason: FinishToolCalls},
	}})
	part, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	if part.ToolCall == nil || !strings.HasPrefix(part.ToolCall.ID, "llmux-generated-") || part.ID != part.ToolCall.ID {
		t.Fatalf("canonical id-less tool call = %#v", part)
	}
}
