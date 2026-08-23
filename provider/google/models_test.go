package google

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/Viking602/llmux"
)

func TestListModelsPaginates(t *testing.T) {
	var pages atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/models" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if request.Header.Get("X-Goog-Api-Key") != "test-key" {
			t.Errorf("api key = %q", request.Header.Get("X-Goog-Api-Key"))
		}
		page := pages.Add(1)
		if page == 1 {
			_, _ = fmt.Fprint(response, `{"models":[{"name":"models/gemini-a","displayName":"A"}],"nextPageToken":"tok-2"}`)
			return
		}
		if request.URL.Query().Get("pageToken") != "tok-2" {
			t.Errorf("pageToken = %q", request.URL.Query().Get("pageToken"))
		}
		_, _ = fmt.Fprint(response, `{"models":[{"name":"gemini-b","displayName":"B"}]}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "gemini-a" || models[1].ID != "gemini-b" || models[0].DisplayName != "A" {
		t.Fatalf("models = %#v", models)
	}
}

func TestParseGoogleModelCapabilitiesDistinguishesGenerationEmbeddingAndUnknown(t *testing.T) {
	models, _, err := parseGoogleModelPage([]byte(`{"models":[
		{"name":"models/gemini","supportedGenerationMethods":["generateContent","streamGenerateContent"],"inputTokenLimit":100,"outputTokenLimit":20},
		{"name":"models/embed","supportedGenerationMethods":["embedContent"]},
		{"name":"models/unknown"}
	]}`))
	if err != nil {
		t.Fatal(err)
	}
	if models[0].Capabilities == nil || models[0].Capabilities.Streaming == nil ||
		!*models[0].Capabilities.Streaming || models[0].Capabilities.OutputModalities[0] != llmux.ModalityText {
		t.Fatalf("generation capabilities = %#v", models[0].Capabilities)
	}
	if models[1].Capabilities == nil || models[1].Capabilities.OutputModalities[0] != llmux.ModalityEmbedding {
		t.Fatalf("embedding capabilities = %#v", models[1].Capabilities)
	}
	if models[2].Capabilities != nil {
		t.Fatalf("unknown capabilities = %#v", models[2].Capabilities)
	}
}
