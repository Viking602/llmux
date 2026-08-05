package codex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Viking602/llmux"
)

func TestSubscriptionGenerateAlwaysStreams(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/backend-api/codex/responses" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer account-token" {
			t.Errorf("authorization = %q", got)
		}
		if got := request.Header.Get("Originator"); got != "test-client" {
			t.Errorf("originator = %q", got)
		}
		if got := request.Header.Get("ChatGPT-Account-Id"); got != "account-1" {
			t.Errorf("account id = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["stream"] != true || body["store"] != false {
			t.Errorf("body = %#v", body)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\",\"output\":[],\"usage\":{}}}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{
		APIKey: "account-token", Mode: ModeSubscription,
		BaseURL: server.URL + "/backend-api/codex", Client: server.Client(),
		ChatGPTAccountID: "account-1", Originator: "test-client",
	})
	if err != nil {
		t.Fatal(err)
	}
	model, err := provider.LanguageModel("gpt-5.2-codex")
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Generate(context.Background(), llmux.Request{
		Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")},
		Options:  llmux.CallOptions{BodyOverrides: json.RawMessage(`{"store":true}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "done" || result.FinishReason != llmux.FinishStop {
		t.Fatalf("result = %#v", result)
	}
}

func TestSubscriptionUnauthorizedMapsTokenExpired(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusUnauthorized)
		_, _ = response.Write([]byte(`{"error":{"message":"expired"}}`))
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "expired", Mode: ModeSubscription, Endpoint: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-5.2-codex")
	_, err = model.Generate(context.Background(), llmux.Request{})
	var providerError *llmux.ProviderError
	if !errors.As(err, &providerError) || providerError.Kind != llmux.ErrorTokenExpired || providerError.StatusCode != http.StatusUnauthorized {
		t.Fatalf("error = %#v", err)
	}
}

func TestRefresh(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.Header.Get("Content-Type") != "application/json" {
			t.Errorf("request = %s, content-type = %q", request.Method, request.Header.Get("Content-Type"))
		}
		var body map[string]string
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["grant_type"] != "refresh_token" || body["refresh_token"] != "refresh-1" || body["client_id"] != "client-1" {
			t.Errorf("body = %#v", body)
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{"access_token":"access-2","refresh_token":"refresh-2","expires_in":3600}`))
	}))
	defer server.Close()

	tokens, err := Refresh(context.Background(), RefreshConfig{
		RefreshToken: "refresh-1", ClientID: "client-1", TokenURL: server.URL, Client: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if tokens.AccessToken != "access-2" || tokens.RefreshToken != "refresh-2" || tokens.ExpiresInSeconds == nil || *tokens.ExpiresInSeconds != 3600 {
		t.Fatalf("tokens = %#v", tokens)
	}
}

func TestRefreshDoesNotRetry(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		requests++
		http.Error(response, "try again", http.StatusInternalServerError)
	}))
	defer server.Close()

	_, err := Refresh(context.Background(), RefreshConfig{
		RefreshToken: "refresh-1", ClientID: "client-1", TokenURL: server.URL, Client: server.Client(),
	})
	if err == nil || requests != 1 {
		t.Fatalf("error = %v, requests = %d", err, requests)
	}
}

func TestRefreshDoesNotFollowRedirect(t *testing.T) {
	replayed := 0
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		replayed++
	}))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		http.Redirect(response, request, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	_, err := Refresh(context.Background(), RefreshConfig{
		RefreshToken: "refresh-1", ClientID: "client-1", TokenURL: source.URL, Client: source.Client(),
	})
	if err == nil || replayed != 0 {
		t.Fatalf("error = %v, replayed = %d", err, replayed)
	}
}

func TestRefreshErrorDoesNotExposeTokenResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"refresh_token":"secret-refresh"}`))
	}))
	defer server.Close()

	_, err := Refresh(context.Background(), RefreshConfig{
		RefreshToken: "refresh-1", ClientID: "client-1", TokenURL: server.URL, Client: server.Client(),
	})
	var providerError *llmux.ProviderError
	if !errors.As(err, &providerError) || len(providerError.Raw) != 0 || strings.Contains(err.Error(), "secret-refresh") {
		t.Fatalf("error = %#v", err)
	}
}
