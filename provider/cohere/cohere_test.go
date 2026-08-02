package cohere

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
