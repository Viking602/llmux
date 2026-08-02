package xai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestDefaultsToResponses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses" {
			t.Errorf("path = %q", request.URL.Path)
		}
		_, _ = response.Write([]byte(`{"status":"completed","output":[],"usage":{}}`))
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "test", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("grok-test")
	if _, err := model.Generate(context.Background(), llmux.Request{}); err != nil {
		t.Fatal(err)
	}
}
