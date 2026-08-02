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
	}
	for _, id := range []string{"openai", "anthropic", "google", "bedrock", "tavily", "groq", "deepseek"} {
		if _, ok := Lookup(id); !ok {
			t.Fatalf("missing provider %s", id)
		}
	}
	for id, want := range map[string]Backend{
		"groq": BackendResponses, "openrouter": BackendResponses, "vercel": BackendResponses,
		"minimax": BackendAnthropic, "tencent_token_plan": BackendAnthropic, "togetherai": BackendOpenAICompat,
	} {
		provider, ok := Lookup(id)
		if !ok || provider.Backend != want {
			t.Fatalf("%s backend = %q/%v, want %q", id, provider.Backend, ok, want)
		}
	}
}
