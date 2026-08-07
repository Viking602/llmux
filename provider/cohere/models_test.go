package cohere

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/models" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if !strings.HasPrefix(request.Header.Get("Authorization"), "Bearer ") {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		_, _ = fmt.Fprint(response, `{"models":[{"name":"command-r","endpoints":["chat"]},{"name":"embed-v4.0"}]}`)
	}))
	defer server.Close()

	// Chat uses /v2; list should rewrite to /v1/models.
	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL + "/v2", Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "command-r" || models[1].ID != "embed-v4.0" {
		t.Fatalf("models = %#v", models)
	}
}
