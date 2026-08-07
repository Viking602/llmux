package bedrock

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodGet {
			t.Errorf("method = %s", request.Method)
		}
		if request.URL.Path != "/foundation-models" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if !strings.Contains(request.Header.Get("Authorization"), "Credential=AKID/") {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		_, _ = fmt.Fprint(response, `{"modelSummaries":[{"modelId":"anthropic.claude-test","modelName":"Claude Test","providerName":"Anthropic"}]}`)
	}))
	defer server.Close()

	provider, err := New(Config{
		Region: "us-east-1", BaseURL: server.URL, Client: server.Client(),
		Credentials: Credentials{AccessKeyID: "AKID", SecretAccessKey: "SECRET"},
		Now:         func() time.Time { return time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "anthropic.claude-test" || models[0].DisplayName != "Claude Test" || models[0].OwnedBy != "Anthropic" {
		t.Fatalf("models = %#v", models)
	}
}

func TestControlPlaneBase(t *testing.T) {
	provider := &Provider{config: Config{Region: "eu-west-1", BaseURL: "https://bedrock-runtime.eu-west-1.amazonaws.com"}}
	if got := provider.controlPlaneBase(); got != "https://bedrock.eu-west-1.amazonaws.com" {
		t.Fatalf("control plane = %q", got)
	}
	provider.config.BaseURL = "http://127.0.0.1:9"
	if got := provider.controlPlaneBase(); got != "http://127.0.0.1:9" {
		t.Fatalf("custom base = %q", got)
	}
}
