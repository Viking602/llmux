package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestListModelsPaginates(t *testing.T) {
	var pages atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/models" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if request.Header.Get("X-Api-Key") != "test-key" {
			t.Errorf("x-api-key = %q", request.Header.Get("X-Api-Key"))
		}
		if request.Header.Get("Anthropic-Version") == "" {
			t.Error("missing anthropic-version")
		}
		page := pages.Add(1)
		switch page {
		case 1:
			if request.URL.Query().Get("after_id") != "" {
				t.Errorf("unexpected after_id on first page")
			}
			_, _ = fmt.Fprint(response, `{"data":[{"id":"claude-a","display_name":"A","created_at":"2024-01-01T00:00:00Z","type":"model"}],"has_more":true,"last_id":"claude-a"}`)
		default:
			if request.URL.Query().Get("after_id") != "claude-a" {
				t.Errorf("after_id = %q", request.URL.Query().Get("after_id"))
			}
			_, _ = fmt.Fprint(response, `{"data":[{"id":"claude-b","display_name":"B","type":"model"}],"has_more":false,"last_id":"claude-b"}`)
		}
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
	if len(models) != 2 || models[0].ID != "claude-a" || models[0].DisplayName != "A" || models[1].ID != "claude-b" {
		t.Fatalf("models = %#v", models)
	}
	if pages.Load() != 2 {
		t.Fatalf("pages = %d", pages.Load())
	}
}
