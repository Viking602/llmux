package bedrock

import (
	"context"
	"encoding/binary"
	"hash/crc32"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Viking602/llmux"
)

func TestGenerateSignsAndParsesConverse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/model/model%3Atest/converse" && request.URL.Path != "/model/model:test/converse" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if !strings.Contains(request.Header.Get("Authorization"), "Credential=AKID/20200102/us-east-1/bedrock/aws4_request") {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		_, _ = response.Write([]byte(`{"output":{"message":{"content":[{"text":"ok"},{"toolUse":{"toolUseId":"c1","name":"lookup","input":{"x":1}}}]}},"stopReason":"tool_use","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}`))
	}))
	defer server.Close()
	provider, err := New(Config{Region: "us-east-1", BaseURL: server.URL, Client: server.Client(), Credentials: Credentials{AccessKeyID: "AKID", SecretAccessKey: "SECRET"}, Now: func() time.Time { return time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC) }})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("model:test")
	result, err := model.Generate(context.Background(), llmux.Request{})
	if err != nil || result.Text != "ok" || len(result.ToolCalls) != 1 || result.Usage.TotalTokens != 3 {
		t.Fatalf("result/error = %#v/%v", result, err)
	}
}

func TestEventStreamFrame(t *testing.T) {
	encoded := encodeEvent("event", "contentBlockDelta", `{"delta":{"text":"ok"}}`)
	event, err := readEvent(strings.NewReader(string(encoded)))
	if err != nil || event.EventType != "contentBlockDelta" || string(event.Payload) != `{"delta":{"text":"ok"}}` {
		t.Fatalf("event/error = %#v/%v", event, err)
	}
}

func encodeEvent(messageType, eventType, payload string) []byte {
	headers := []byte{}
	for _, pair := range [][2]string{{":message-type", messageType}, {":event-type", eventType}, {":content-type", "application/json"}} {
		headers = append(headers, byte(len(pair[0])))
		headers = append(headers, pair[0]...)
		headers = append(headers, 7, byte(len(pair[1])>>8), byte(len(pair[1])))
		headers = append(headers, pair[1]...)
	}
	total := 16 + len(headers) + len(payload)
	frame := make([]byte, total)
	binary.BigEndian.PutUint32(frame[:4], uint32(total))
	binary.BigEndian.PutUint32(frame[4:8], uint32(len(headers)))
	binary.BigEndian.PutUint32(frame[8:12], crc32.ChecksumIEEE(frame[:8]))
	copy(frame[12:], headers)
	copy(frame[12+len(headers):], payload)
	binary.BigEndian.PutUint32(frame[total-4:], crc32.ChecksumIEEE(frame[:total-4]))
	return frame
}

func TestToolBlockFinalizationIsIdempotent(t *testing.T) {
	stream := &converseStream{blocks: make(map[int]*blockBuilder)}
	events := []eventMessage{
		{EventType: "contentBlockStart", Payload: []byte(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"","name":"lookup"}}}`)},
		{EventType: "contentBlockDelta", Payload: []byte(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"x\":1}"}}}`)},
		{EventType: "contentBlockStop", Payload: []byte(`{"contentBlockIndex":0}`)},
	}
	for _, event := range events[:2] {
		if err := stream.mapEvent(event); err != nil {
			t.Fatal(err)
		}
	}
	for range 2 {
		if err := stream.mapEvent(events[2]); err != nil {
			t.Fatal(err)
		}
	}
	counts := make(map[llmux.PartKind]int)
	for _, part := range stream.pending {
		counts[part.Kind]++
	}
	for _, kind := range []llmux.PartKind{llmux.PartToolInputStart, llmux.PartToolInputDelta, llmux.PartToolInputEnd, llmux.PartToolCall} {
		if counts[kind] != 1 {
			t.Fatalf("%s parts = %d, want 1", kind, counts[kind])
		}
	}
}

func TestContentBlockIndexCannotBeRebound(t *testing.T) {
	stream := &converseStream{blocks: make(map[int]*blockBuilder)}
	start := eventMessage{EventType: "contentBlockStart", Payload: []byte(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call-1","name":"lookup"}}}`)}
	if err := stream.mapEvent(start); err != nil {
		t.Fatal(err)
	}
	if err := stream.mapEvent(start); err == nil {
		t.Fatal("Bedrock content block index was rebound")
	}
}
