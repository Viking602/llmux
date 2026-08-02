package tavily

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestSearch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/search" || request.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("unexpected request: %s %q", request.URL.Path, request.Header.Get("Authorization"))
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["query"] != "go sdk" || body["include_raw_contents"] != true {
			t.Fatalf("unexpected body: %#v", body)
		}
		_, _ = writer.Write([]byte(`{"answer":"Go SDK","results":[{"title":"docs","url":"https://example.com","score":0.9}]}`))
	}))
	defer server.Close()

	model, err := New(Config{APIKey: "test-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	includeRaw := true
	result, err := model.Search(context.Background(), llmux.SearchRequest{Query: "go sdk", IncludeRawContent: &includeRaw})
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "Go SDK" || len(result.Results) != 1 || result.Results[0].Score != 0.9 {
		t.Fatalf("unexpected result: %#v", result)
	}
}
