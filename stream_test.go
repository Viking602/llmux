package llmux

import (
	"io"
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
