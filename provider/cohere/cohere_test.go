package cohere

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestGenerateAndStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
		}
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(response, "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"r1\"}\n\n")
			_, _ = fmt.Fprint(response, "event: content-start\ndata: {\"type\":\"content-start\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\"}}}}\n\n")
			_, _ = fmt.Fprint(response, "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"ok\"}}}}\n\n")
			_, _ = fmt.Fprint(response, "event: content-end\ndata: {\"type\":\"content-end\",\"index\":0}\n\n")
			_, _ = fmt.Fprint(response, "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":2,\"output_tokens\":1}}}}\n\n")
			return
		}
		_, _ = fmt.Fprint(response, `{"generation_id":"r1","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":2,"output_tokens":1}}}`)
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("command-test")
	result, err := model.Generate(context.Background(), llmux.Request{})
	if err != nil || result.Text != "ok" || result.Reasoning != "hmm" {
		t.Fatalf("result/error = %#v/%v", result, err)
	}
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	result, err = llmux.Collect(stream)
	if err != nil || result.Text != "ok" || result.Usage.TotalTokens != 3 {
		t.Fatalf("stream result/error = %#v/%v", result, err)
	}
}

func TestRepeatedToolCallEndEmitsOnce(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"r1\"}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":1}\"}}}}}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-end\ndata: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-end\ndata: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(response, "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"TOOL_CALL\"}}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("command-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	counts := make(map[llmux.PartKind]int)
	for {
		part, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		counts[part.Kind]++
	}
	for _, kind := range []llmux.PartKind{llmux.PartToolInputStart, llmux.PartToolInputDelta, llmux.PartToolInputEnd, llmux.PartToolCall} {
		if counts[kind] != 1 {
			t.Fatalf("%s parts = %d, want 1", kind, counts[kind])
		}
	}
}

func TestParallelToolIndexesRemainDistinct(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"call-1\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}}}}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":1,\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"call-2\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}}}}}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-end\ndata: {\"type\":\"tool-call-end\",\"index\":0}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-end\ndata: {\"type\":\"tool-call-end\",\"index\":1}\n\n")
		_, _ = fmt.Fprint(response, "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"TOOL_CALL\"}}\n\n")
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("command-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 2 || result.ToolCalls[0].ID != "call-1" || result.ToolCalls[1].ID != "call-2" {
		t.Fatalf("parallel tool calls = %#v", result.ToolCalls)
	}
}

func TestToolEventAfterMessageEndFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n")
		_, _ = fmt.Fprint(response, "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}}}}\n\n")
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("command-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := llmux.Collect(stream); err == nil {
		t.Fatal("Cohere tool event after message-end was accepted")
	}
}
