package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestGenerateAndStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/messages" || request.Header.Get("X-Api-Key") != "test-key" {
			t.Errorf("path/key = %q/%q", request.URL.Path, request.Header.Get("X-Api-Key"))
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(response, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-1\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n")
			_, _ = fmt.Fprint(response, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
			_, _ = fmt.Fprint(response, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n")
			_, _ = fmt.Fprint(response, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
			_, _ = fmt.Fprint(response, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n")
			_, _ = fmt.Fprint(response, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			return
		}
		_, _ = fmt.Fprint(response, `{"id":"msg-1","model":"claude-test","content":[{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("claude-test")
	request := llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")}}
	result, err := model.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.Reasoning != "hmm" || !json.Valid(result.ProviderState) {
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

func TestCustomAuthHeader(t *testing.T) {
	provider, err := New(Config{APIKey: "test-key", APIKeyHeader: "Authorization", APIKeyPrefix: "Bearer "})
	if err != nil {
		t.Fatal(err)
	}
	languageModel, _ := provider.LanguageModel("test")
	headers := languageModel.(*model).headers(nil)
	if headers.Get("Authorization") != "Bearer test-key" || headers.Get("X-Api-Key") != "" {
		t.Fatalf("headers = %#v", headers)
	}
}

func TestMaxTokensDefaultsAndOverrides(t *testing.T) {
	var got []int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		maxTokens, _ := body["max_tokens"].(float64)
		got = append(got, int(maxTokens))
		_, _ = fmt.Fprint(response, `{"id":"msg-1","model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	// Package default when neither Config nor request sets a limit.
	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("claude-test")
	if _, err = model.Generate(context.Background(), llmux.Request{
		Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")},
	}); err != nil {
		t.Fatal(err)
	}

	// Provider-level default for catalog max-output (e.g. DeepSeek 384000).
	provider, err = New(Config{
		APIKey: "test-key", BaseURL: server.URL, Client: server.Client(),
		DefaultMaxOutputTokens: 384000,
	})
	if err != nil {
		t.Fatal(err)
	}
	model, _ = provider.LanguageModel("claude-test")
	if _, err = model.Generate(context.Background(), llmux.Request{
		Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")},
	}); err != nil {
		t.Fatal(err)
	}

	// Per-request MaxOutputTokens always wins over the provider default.
	maxOutput := 2048
	if _, err = model.Generate(context.Background(), llmux.Request{
		Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")},
		Options:  llmux.CallOptions{MaxOutputTokens: &maxOutput},
	}); err != nil {
		t.Fatal(err)
	}

	if len(got) != 3 || got[0] != DefaultMaxOutputTokens || got[1] != 384000 || got[2] != 2048 {
		t.Fatalf("max_tokens sequence = %v, want [%d 384000 2048]", got, DefaultMaxOutputTokens)
	}
}
