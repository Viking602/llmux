package azure

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestAzureEndpointAndAuth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/openai/responses" || request.URL.Query().Get("api-version") != "v-test" || request.Header.Get("api-key") != "secret" {
			t.Errorf("request = %s, headers = %#v", request.URL.String(), request.Header)
		}
		_, _ = response.Write([]byte(`{"id":"1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "secret", ResourceURL: server.URL, Deployment: "deploy", APIVersion: "v-test", Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("deploy")
	result, err := model.Generate(context.Background(), llmux.Request{})
	if err != nil || result.Text != "ok" {
		t.Fatalf("result/error = %#v/%v", result, err)
	}
}
