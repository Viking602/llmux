package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Viking602/llmux"
)

func TestChatGenerateAndStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["model"] != "gpt-test" {
			t.Errorf("model = %#v", body["model"])
		}
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(response, "data: {\"id\":\"chat-1\",\"model\":\"gpt-test\",\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n")
			_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = fmt.Fprint(response, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n")
			_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
			return
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(response, `{"id":"chat-1","model":"gpt-test","created":1,"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client(), WireAPI: ChatCompletions})
	if err != nil {
		t.Fatal(err)
	}
	model, err := provider.LanguageModel("gpt-test")
	if err != nil {
		t.Fatal(err)
	}
	request := llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")}}
	result, err := model.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.FinishReason != llmux.FinishStop || result.Usage.TotalTokens != 3 {
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
	if result.Text != "hello" || result.FinishReason != llmux.FinishStop || result.Usage.TotalTokens != 3 {
		t.Fatalf("stream result = %#v", result)
	}
}

func TestResponsesPreservesProviderStateAndToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses" {
			t.Errorf("path = %q", request.URL.Path)
		}
		_, _ = fmt.Fprint(response, `{"id":"resp-1","model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg-1","content":[{"type":"output_text","text":"done"}]},{"type":"function_call","id":"item-1","call_id":"call-1","name":"lookup","arguments":"{\"id\":7}"}],"usage":{"prompt_tokens":4,"completion_tokens":3}}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	result, err := model.Generate(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "done" || len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "call-1" || !json.Valid(result.ProviderState) || result.Usage.TotalTokens != 7 {
		t.Fatalf("result = %#v", result)
	}
}

func TestProtectedBodyFieldsCannotBeOverridden(t *testing.T) {
	provider, err := New(Config{APIKey: "test-key"})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	_, err = model.Generate(context.Background(), llmux.Request{Options: llmux.CallOptions{BodyOverrides: json.RawMessage(`{"model":"stolen"}`)}})
	if err == nil {
		t.Fatal("expected protected body override error")
	}
}

func TestResponsesStreamPreservesProviderFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"error\",\"code\":\"server_is_overloaded\",\"message\":\"retry me\"}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", Endpoint: server.URL, Client: server.Client(), WireAPI: Responses})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	part, err := stream.Recv()
	if err != nil || part.Kind != llmux.PartError || part.Err == nil || !strings.Contains(part.Err.Error(), "retry me") {
		t.Fatalf("part=%#v error=%v", part, err)
	}
}

func TestResponsesStreamAcceptsEOFAfterCompleted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\",\"output\":[],\"usage\":{}}}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", Endpoint: server.URL, Client: server.Client(), WireAPI: Responses})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil || result.Text != "done" || result.FinishReason != llmux.FinishStop {
		t.Fatalf("result=%#v error=%v", result, err)
	}
}

func TestResponsesUsagePreservesReportedZero(t *testing.T) {
	result, err := parseResponsesResult([]byte(`{"status":"completed","output":[],"usage":{"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Usage.CachedInputTokensReported || !result.Usage.CacheWriteInputTokensReported {
		t.Fatalf("usage=%#v", result.Usage)
	}
}
