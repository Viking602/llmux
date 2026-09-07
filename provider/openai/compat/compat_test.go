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
	if !ok || deepSeek.Behavior != BehaviorDeepSeek || deepSeek.Protocol != ProtocolAnthropic || deepSeek.BaseURL != "https://api.deepseek.com/anthropic" {
		t.Fatalf("deepseek = %#v/%v", deepSeek, ok)
	}
	if minimax, ok := Lookup("minimax"); !ok || minimax.Protocol != ProtocolAnthropic {
		t.Fatalf("minimax = %#v/%v", minimax, ok)
	}
	for id, baseURL := range map[string]string{
		"kimi":            "https://api.moonshot.ai/anthropic",
		"kimi-for-coding": "https://api.kimi.com/coding",
		"moonshotai":      "https://api.moonshot.cn/anthropic",
		"mimo":            "https://api.xiaomimimo.com/anthropic",
		"longcat":         "https://api.longcat.chat/anthropic",
		"zai":             "https://api.z.ai/api/anthropic",
		"zhipu-v4":        "https://open.bigmodel.cn/api/anthropic",
		"siliconflow":     "https://api.siliconflow.cn",
	} {
		profile, ok := Lookup(id)
		if !ok || profile.Protocol != ProtocolAnthropic || profile.BaseURL != baseURL {
			t.Fatalf("%s = %#v/%v, want anthropic %s", id, profile, ok, baseURL)
		}
	}
	// Underscore aliases keep working for older callers.
	if profile, ok := Lookup("alibaba_coding_plan"); !ok || profile.ID != "alibaba-coding-plan" {
		t.Fatalf("underscore alias = %#v/%v", profile, ok)
	}
	if inferenceHub, ok := Lookup("inferencehub"); !ok || inferenceHub.BaseURL != "https://app.inferencehub.tech/v1" || inferenceHub.EnvKey != "INFERENCEHUB_API_KEY" {
		t.Fatalf("inferencehub = %#v/%v", inferenceHub, ok)
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
	provider, err := New("zai-coding-plan", Config{APIKey: "test", BaseURL: server.URL, Client: server.Client()})
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

func TestNovitaUsesOpenAIFamilyAndAnthropic(t *testing.T) {
	profile, ok := Lookup("novita")
	if !ok || profile.BaseURL != "https://api.novita.ai/openai/v1" {
		t.Fatalf("novita = %#v/%v", profile, ok)
	}
	want := []Protocol{ProtocolChatCompletions, ProtocolResponses, ProtocolAnthropic}
	got := profile.Protocols()
	if len(got) != len(want) {
		t.Fatalf("protocols = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("protocols = %#v, want %#v", got, want)
		}
	}
	if profile.baseURLFor(ProtocolAnthropic) != "https://api.novita.ai/anthropic" {
		t.Fatalf("anthropic base = %q", profile.baseURLFor(ProtocolAnthropic))
	}
	if profile.baseURLFor(ProtocolResponses) != profile.BaseURL {
		t.Fatalf("responses base = %q", profile.baseURLFor(ProtocolResponses))
	}

	var saw []string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		saw = append(saw, request.URL.Path)
		switch {
		case strings.HasSuffix(request.URL.Path, "/chat/completions"):
			_, _ = response.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"chat"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
		case strings.HasSuffix(request.URL.Path, "/responses"):
			_, _ = response.Write([]byte(`{"id":"resp-1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"resp"}]}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
		case strings.HasSuffix(request.URL.Path, "/messages"):
			_, _ = response.Write([]byte(`{"id":"msg-1","model":"glm","content":[{"type":"text","text":"anth"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		default:
			t.Errorf("unexpected path %q", request.URL.Path)
		}
	}))
	defer server.Close()

	for _, tc := range []struct {
		protocol Protocol
		wantPath string
		wantText string
	}{
		{ProtocolChatCompletions, "/chat/completions", "chat"},
		{ProtocolResponses, "/responses", "resp"},
		{ProtocolAnthropic, "/v1/messages", "anth"},
	} {
		saw = nil
		provider, err := New("novita", Config{APIKey: "test", BaseURL: server.URL, Client: server.Client(), Protocol: tc.protocol})
		if err != nil {
			t.Fatalf("%s new: %v", tc.protocol, err)
		}
		model, err := provider.LanguageModel("zai-org/glm-5.3-flash")
		if err != nil {
			t.Fatalf("%s model: %v", tc.protocol, err)
		}
		result, err := model.Generate(context.Background(), llmux.Request{})
		if err != nil || result.Text != tc.wantText {
			t.Fatalf("%s result/error = %#v/%v", tc.protocol, result, err)
		}
		if len(saw) == 0 || !strings.HasSuffix(saw[0], tc.wantPath) {
			t.Fatalf("%s path = %#v, want suffix %q", tc.protocol, saw, tc.wantPath)
		}
	}

	if _, err := New("novita", Config{APIKey: "test", Protocol: "soap"}); err == nil {
		t.Fatal("expected unsupported protocol error")
	}
}
