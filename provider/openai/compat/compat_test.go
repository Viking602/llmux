package compat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/Viking602/llmux"
)

func TestRegistryAndProfiles(t *testing.T) {
	groq, ok := Lookup("groq")
	if !ok || groq.Behavior != BehaviorGroq || groq.Protocol != ProtocolResponses {
		t.Fatalf("groq = %#v/%v", groq, ok)
	}
	deepSeek, ok := Lookup("deepseek")
	if !ok || deepSeek.Behavior != BehaviorDeepSeek {
		t.Fatalf("deepseek = %#v/%v", deepSeek, ok)
	}
	if minimax, ok := Lookup("minimax"); !ok || minimax.Protocol != ProtocolAnthropic {
		t.Fatalf("minimax = %#v/%v", minimax, ok)
	}
	for _, profile := range All() {
		if profile.BaseURL == "" {
			continue
		}
		baseURL := strings.ReplaceAll(profile.BaseURL, "{CLOUDFLARE_ACCOUNT_ID}", "account")
		parsed, err := url.Parse(baseURL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			t.Fatalf("%s base URL = %q", profile.ID, profile.BaseURL)
		}
	}
}

func TestGroqUsesResponsesAndOmitsTopK(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses" {
			t.Errorf("path = %q", request.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if _, exists := body["top_k"]; exists {
			t.Error("groq request contains top_k")
		}
		_, _ = response.Write([]byte(`{"id":"resp-1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"prompt_tokens":2,"completion_tokens":1}}`))
	}))
	defer server.Close()
	topK := 10
	provider, err := New("groq", Config{APIKey: "test", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("test")
	result, err := model.Generate(context.Background(), llmux.Request{Options: llmux.CallOptions{TopK: &topK}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.TotalTokens != 3 || len(result.Warnings) != 1 {
		t.Fatalf("result = %#v", result)
	}
}

func TestAnthropicProfileUsesProviderAuth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/messages" || request.Header.Get("Authorization") != "Bearer test" || request.Header.Get("X-Api-Key") != "" {
			t.Errorf("request = %s, headers = %#v", request.URL.String(), request.Header)
		}
		_, _ = response.Write([]byte(`{"id":"msg-1","model":"glm-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()
	provider, err := New("zai_coding_plan", Config{APIKey: "test", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("glm-test")
	result, err := model.Generate(context.Background(), llmux.Request{})
	if err != nil || result.Text != "ok" {
		t.Fatalf("result/error = %#v/%v", result, err)
	}
}

func TestLocalResponsesAllowsEmptyAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses" {
			t.Errorf("path = %q", request.URL.Path)
		}
		_, _ = response.Write([]byte(`{"status":"completed","output":[],"usage":{}}`))
	}))
	defer server.Close()
	provider, err := New("ollama", Config{BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("local")
	if _, err := model.Generate(context.Background(), llmux.Request{}); err != nil {
		t.Fatal(err)
	}
}

func TestCloudflareRequiresAccountIDForDefaultURL(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	if _, err := New("cloudflare", Config{APIKey: "test"}); err == nil {
		t.Fatal("expected missing account ID error")
	}
}

func TestProfileWithoutDefaultURLRequiresOverride(t *testing.T) {
	if _, err := New("snowflake", Config{APIKey: "test"}); err == nil {
		t.Fatal("expected explicit base URL error")
	}
}
