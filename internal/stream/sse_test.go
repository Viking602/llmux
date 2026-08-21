package stream

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestSSEReaderParsesMultilineFrame(t *testing.T) {
	reader := NewSSEReader(strings.NewReader(": keepalive\r\nevent: delta\r\nid: 7\r\ndata: one\r\ndata: two\r\n\r\n"), 0)
	event, err := reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if event.Name != "delta" || event.ID != "7" || event.Data != "one\ntwo" || event.Comment != "keepalive" {
		t.Fatalf("event = %#v", event)
	}
}

func TestSSEReaderRejectsTruncatedFrame(t *testing.T) {
	reader := NewSSEReader(strings.NewReader("data: {\"partial\":true}"), 0)
	_, err := reader.Next()
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("error = %v", err)
	}
}

func TestSSEReaderBoundsFrame(t *testing.T) {
	reader := NewSSEReader(strings.NewReader("data: 123456\n\n"), 5)
	if _, err := reader.Next(); err == nil {
		t.Fatal("expected frame size error")
	}
}

func TestSSEReaderRejectsOversizedUnterminatedLine(t *testing.T) {
	reader := NewSSEReader(strings.NewReader("data: "+strings.Repeat("x", 65)), 64)
	if _, err := reader.Next(); err == nil || !strings.Contains(err.Error(), "SSE line exceeds 64 bytes") {
		t.Fatalf("error = %v", err)
	}
}
