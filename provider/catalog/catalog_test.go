package catalog

import "testing"

func TestProviderMatrix(t *testing.T) {
	all := All()
	if len(all) == 0 || len(Explicit()) == 0 {
		t.Fatal("provider catalog is empty")
	}
	seen := make(map[string]bool, len(all))
	for _, provider := range all {
		if provider.ID == "" || provider.Backend == "" || len(provider.Capabilities) == 0 {
			t.Fatalf("incomplete provider: %#v", provider)
		}
		if seen[provider.ID] {
			t.Fatalf("duplicate provider: %s", provider.ID)
		}
		seen[provider.ID] = true
		descriptor := provider.Descriptor()
		if descriptor.Name != provider.ID || len(descriptor.WireProtocols) != 1 ||
			len(descriptor.Capabilities) != len(provider.Capabilities) {
			t.Fatalf("portable descriptor drift for %s: %#v", provider.ID, descriptor)
		}
	}
	for _, id := range []string{"openai", "anthropic", "google", "bedrock", "codex", "inferencehub", "tavily", "groq", "deepseek"} {
		if _, ok := Lookup(id); !ok {
			t.Fatalf("missing provider %s", id)
		}
	}
	for id, want := range map[string]Backend{
		"codex": BackendResponses, "groq": BackendResponses, "openrouter": BackendResponses, "vercel": BackendResponses,
		"inferencehub": BackendOpenAICompat, "minimax": BackendAnthropic, "tencent-token-plan": BackendAnthropic,
		"deepseek": BackendAnthropic, "kimi": BackendAnthropic, "mimo": BackendAnthropic, "zai": BackendAnthropic,
		"togetherai": BackendOpenAICompat, "alibaba-coding-plan": BackendAnthropic,
		"black-forest-labs": BackendNativeHTTP, "vertex-ai-anthropic-models": BackendOpenAICompat,
	} {
		provider, ok := Lookup(id)
		if !ok || provider.Backend != want {
			t.Fatalf("%s backend = %q/%v, want %q", id, provider.Backend, ok, want)
		}
	}
	if provider, ok := Lookup("tencent_token_plan"); !ok || provider.ID != "tencent-token-plan" {
		t.Fatalf("underscore alias = %#v/%v", provider, ok)
	}
	for _, id := range []string{"openai", "anthropic", "google", "bedrock", "cohere", "mistral", "xai", "groq", "deepseek", "ollama"} {
		provider, ok := Lookup(id)
		if !ok || !hasCapability(provider, ListModels) {
			t.Fatalf("%s should advertise list_models capability", id)
		}
	}
	for _, id := range []string{"voyage", "tavily", "codex", "vertex"} {
		provider, ok := Lookup(id)
		if !ok || hasCapability(provider, ListModels) {
			t.Fatalf("%s should not advertise list_models capability", id)
		}
	}
}

func hasCapability(provider Provider, want Capability) bool {
	for _, capability := range provider.Capabilities {
		if capability == want {
			return true
		}
	}
	return false
}
