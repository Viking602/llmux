package google

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
