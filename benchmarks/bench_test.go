package benchmarks

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/openai"
)

func BenchmarkOpenAIChatGenerate(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(response, `{"id":"chat-1","model":"bench","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	defer server.Close()
	provider, _ := openai.New(openai.Config{APIKey: "bench", BaseURL: server.URL, Client: server.Client(), Retry: llmux.RetryPolicy{MaxAttempts: 1}, WireAPI: openai.ChatCompletions})
	model, _ := provider.LanguageModel("bench")
	request := llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")}}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := model.Generate(ctx, request); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOpenAIChatStream64Chunks(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		for range 64 {
			_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n")
		}
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	provider, _ := openai.New(openai.Config{APIKey: "bench", BaseURL: server.URL, Client: server.Client(), Retry: llmux.RetryPolicy{MaxAttempts: 1}, WireAPI: openai.ChatCompletions})
	model, _ := provider.LanguageModel("bench")
	request := llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")}}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		stream, err := model.Stream(ctx, request)
		if err != nil {
			b.Fatal(err)
		}
		if _, err = llmux.Collect(stream); err != nil {
			b.Fatal(err)
		}
	}
}
