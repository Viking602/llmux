package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodGet {
			t.Errorf("method = %s", request.Method)
		}
		if request.URL.Path != "/models" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		_, _ = fmt.Fprint(response, `{"object":"list","data":[{"id":"gpt-test","object":"model","created":1,"owned_by":"openai"},{"id":"embed-test","object":"model","created":2,"owned_by":"openai"}]}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client(), WireAPI: ChatCompletions})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "gpt-test" || models[0].OwnedBy != "openai" || models[1].ID != "embed-test" {
		t.Fatalf("models = %#v", models)
	}
	viaHelper, err := llmux.ListModels(context.Background(), provider)
	if err != nil || len(viaHelper) != 2 {
		t.Fatalf("helper = %#v / %v", viaHelper, err)
	}
}

func TestListModelsCustomURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/openai/models" || request.URL.RawQuery != "api-version=2024-10-21" {
			t.Errorf("url = %s?%s", request.URL.Path, request.URL.RawQuery)
		}
		if request.Header.Get("api-key") != "azure-key" {
			t.Errorf("api-key = %q", request.Header.Get("api-key"))
		}
		_, _ = fmt.Fprint(response, `{"data":[{"id":"gpt-4o","created_at":99}]}`)
	}))
	defer server.Close()

	provider, err := New(Config{
		APIKey: "azure-key", BaseURL: server.URL, Client: server.Client(), WireAPI: ChatCompletions,
		APIKeyHeader: "api-key", ListModelsURL: server.URL + "/openai/models?api-version=2024-10-21",
		ProviderName: "azure",
	})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "gpt-4o" || models[0].Created != 99 {
		t.Fatalf("models = %#v", models)
	}
}
