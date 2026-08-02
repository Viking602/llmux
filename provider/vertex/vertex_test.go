package vertex

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestVertexBuildsPublisherEndpointAndRefreshesToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		want := "/projects/project/locations/europe/publishers/google/models/gemini-test:generateContent"
		if request.URL.Path != want || request.Header.Get("Authorization") != "Bearer refreshed" {
			t.Errorf("path/auth = %q/%q", request.URL.Path, request.Header.Get("Authorization"))
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	defer server.Close()
	provider, err := New(Config{Project: "project", Location: "europe", BaseURL: server.URL, Client: server.Client(), TokenSource: func(context.Context) (string, error) { return "refreshed", nil }})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gemini-test")
	result, err := model.Generate(context.Background(), llmux.Request{})
	if err != nil || result.Text != "ok" {
		t.Fatalf("result/error = %#v/%v", result, err)
	}
}
