package google

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
		if request.Header.Get("X-Goog-Api-Key") != "key" {
			t.Errorf("key = %q", request.Header.Get("X-Goog-Api-Key"))
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if _, ok := body["contents"]; !ok {
			t.Error("missing contents")
		}
		payload := `{"responseId":"r1","modelVersion":"gemini-test","candidates":[{"content":{"parts":[{"text":"hello"},{"functionCall":{"id":"c1","name":"lookup","args":{"x":1}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}`
		if request.URL.Query().Get("alt") == "sse" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(response, "data: %s\n\n", payload)
			return
		}
		_, _ = fmt.Fprint(response, payload)
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gemini-test")
	request := llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")}}
	result, err := model.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || len(result.ToolCalls) != 1 || result.FinishReason != llmux.FinishToolCalls {
		t.Fatalf("result = %#v", result)
	}
	stream, err := model.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	result, err = llmux.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || len(result.ToolCalls) != 1 || result.Usage.TotalTokens != 3 {
		t.Fatalf("stream result = %#v", result)
	}
}

func TestRepeatedFunctionCallIDEmitsOnce(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"responseId\":\"r1\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call-1\",\"name\":\"lookup\",\"args\":{\"x\":1}}}]}}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"responseId\":\"r1\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call-1\",\"name\":\"lookup\",\"args\":{\"x\":1}}}]},\"finishReason\":\"STOP\"}]}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gemini-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "call-1" {
		t.Fatalf("tool calls = %#v", result.ToolCalls)
	}
}

func TestCandidateAfterTerminalFinishFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call-1\",\"name\":\"lookup\",\"args\":{}}}]}}]}\n\n")
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gemini-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := llmux.Collect(stream); err == nil {
		t.Fatal("Google candidate after terminal finish was accepted")
	}
}

func TestStreamErrorDiscardsQueuedToolParts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call-1\",\"name\":\"lookup\",\"args\":{\"x\":1}}},{\"functionCall\":{\"id\":\"call-1\",\"name\":\"lookup\",\"args\":{\"x\":2}}}]}}]}\n\n")
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gemini-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	part, err := stream.Recv()
	if err != nil || part.Kind != llmux.PartError {
		t.Fatalf("error part=%#v error=%v", part, err)
	}
	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("queued tool parts survived terminal error: %v", err)
	}
}
