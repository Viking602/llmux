// Package catalog exposes the providers supported by llmux.
package catalog

import (
	"sort"
	"strings"

	opencompat "github.com/Viking602/llmux/provider/openai/compat"
)

type Capability string

const (
	Language      Capability = "language"
	Embedding     Capability = "embedding"
	Reranking     Capability = "reranking"
	Speech        Capability = "speech"
	Transcription Capability = "transcription"
	Image         Capability = "image"
	Video         Capability = "video"
	Search        Capability = "search"
	Files         Capability = "files"
	// ListModels marks providers with a programmatic models discovery API.
	// Presence is optimistic for openai-compatible gateways; individual
	// endpoints may still return 404.
	ListModels Capability = "list_models"
)

type Backend string

const (
	BackendOpenAICompat Backend = "openai-compatible"
	BackendOpenAI       Backend = "openai"
	BackendAnthropic    Backend = "anthropic"
	BackendAzure        Backend = "azure"
	BackendBedrock      Backend = "bedrock"
	BackendCohere       Backend = "cohere"
	BackendGoogle       Backend = "google"
	BackendMistral      Backend = "mistral"
	BackendVertex       Backend = "vertex"
	BackendVoyage       Backend = "voyage"
	BackendXAI          Backend = "xai"
	BackendResponses    Backend = "open-responses"
	BackendNativeHTTP   Backend = "native-http"
)

type Provider struct {
	ID           string       `json:"id"`
	Backend      Backend      `json:"backend"`
	Capabilities []Capability `json:"capabilities"`
	Generated    bool         `json:"generated,omitempty"`
}

func Lookup(id string) (Provider, bool) {
	id = normalizeProviderID(id)
	if profile, ok := opencompat.Lookup(id); ok {
		return providerFromProfile(profile), true
	}
	provider, ok := explicitByID[id]
	return provider, ok
}

// normalizeProviderID lowercases ids and maps underscores to hyphens so
// catalog lookups align with models.dev-style provider ids.
func normalizeProviderID(id string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(id)), "_", "-")
}

func All() []Provider {
	byID := make(map[string]Provider, len(explicit)+len(opencompat.All()))
	for _, provider := range explicit {
		byID[provider.ID] = provider
	}
	for _, profile := range opencompat.All() {
		byID[profile.ID] = providerFromProfile(profile)
	}
	result := make([]Provider, 0, len(byID))
	for _, provider := range byID {
		result = append(result, provider)
	}
	sort.Slice(result, func(left, right int) bool { return result[left].ID < result[right].ID })
	return result
}

func Explicit() []Provider { return append([]Provider(nil), explicit...) }

func p(id string, backend Backend, capabilities ...Capability) Provider {
	return Provider{ID: id, Backend: backend, Capabilities: capabilities}
}

func providerFromProfile(profile opencompat.Profile) Provider {
	backend := BackendOpenAICompat
	switch profile.Protocol {
	case opencompat.ProtocolResponses:
		backend = BackendResponses
	case opencompat.ProtocolAnthropic:
		backend = BackendAnthropic
	}
	capabilities := []Capability{Language, ListModels}
	return Provider{ID: profile.ID, Backend: backend, Capabilities: capabilities, Generated: true}
}

var explicit = []Provider{
	p("anthropic", BackendAnthropic, Language, ListModels),
	p("anthropic-aws", BackendBedrock, Language, ListModels),
	p("azure", BackendAzure, Language, ListModels),
	p("bedrock", BackendBedrock, Language, Embedding, Reranking, Image, ListModels),
	p("cohere", BackendCohere, Language, Embedding, Reranking, ListModels),
	p("codex", BackendResponses, Language),
	p("google", BackendGoogle, Language, Embedding, Image, Video, Files, ListModels),
	p("mistral", BackendMistral, Language, Embedding, ListModels),
	p("openai", BackendOpenAI, Language, Embedding, Speech, Transcription, Image, Files, ListModels),
	p("vertex", BackendVertex, Language, Embedding, Transcription, Image, Video),
	p("voyage", BackendVoyage, Embedding, Reranking),
	p("openrouter", BackendOpenAICompat, Language, ListModels),
	p("xai", BackendXAI, Language, ListModels),
	p("huggingface", BackendOpenAICompat, Language, ListModels),
	p("llamafile", BackendOpenAICompat, Language, ListModels),
	p("lmstudio", BackendOpenAICompat, Language, ListModels),
	p("mistralrs", BackendOpenAICompat, Language, ListModels),
	p("ollama", BackendOpenAICompat, Language, ListModels),
	p("cartesia", BackendNativeHTTP, Speech, Transcription),
	p("elevenlabs", BackendNativeHTTP, Speech, Transcription),
	p("hume", BackendNativeHTTP, Speech),
	p("lmnt", BackendNativeHTTP, Speech),
	p("assemblyai", BackendNativeHTTP, Transcription),
	p("deepgram", BackendNativeHTTP, Transcription),
	p("fal", BackendNativeHTTP, Transcription, Image, Video),
	p("gladia", BackendNativeHTTP, Transcription),
	p("revai", BackendNativeHTTP, Transcription),
	p("black-forest-labs", BackendNativeHTTP, Image),
	p("luma", BackendNativeHTTP, Image),
	p("prodia", BackendNativeHTTP, Image, Video),
	p("replicate", BackendNativeHTTP, Image, Video),
	p("klingai", BackendNativeHTTP, Video),
	p("open-responses", BackendResponses, Language, ListModels),
	p("cybertron", BackendOpenAICompat, Language, ListModels),
	p("docker-model-runner", BackendOpenAICompat, Language, ListModels),
	p("gaudi", BackendOpenAICompat, Language, ListModels),
	p("jlama", BackendOpenAICompat, Language, ListModels),
	p("litellm-proxy", BackendOpenAICompat, Language, ListModels),
	p("llamacpp", BackendOpenAICompat, Language, ListModels),
	p("local", BackendOpenAICompat, Language, ListModels),
	p("localai", BackendOpenAICompat, Language, ListModels),
	p("mlx", BackendOpenAICompat, Language, ListModels),
	p("omlx", BackendOpenAICompat, Language, ListModels),
	p("onnx", BackendOpenAICompat, Language, ListModels),
	p("oobabooba", BackendOpenAICompat, Language, ListModels),
	p("openvino", BackendOpenAICompat, Language, ListModels),
	p("sglang", BackendOpenAICompat, Language, ListModels),
	p("vllm", BackendOpenAICompat, Language, ListModels),
	p("xinference", BackendOpenAICompat, Language, ListModels),
	p("jina-ai", BackendNativeHTTP, Reranking),
	p("aws-polly", BackendNativeHTTP, Speech),
	p("recraft", BackendNativeHTTP, Image),
	p("stability", BackendNativeHTTP, Image),
	p("runwayml", BackendNativeHTTP, Video),
	p("bedrock-mantle", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-ai21-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-anthropic-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-deepseek-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-llama-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-minimax-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-mistral-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-moonshot-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-openai-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-qwen-models", BackendOpenAICompat, Language, ListModels),
	p("vertex-ai-zai-models", BackendOpenAICompat, Language, ListModels),
	p("dataforseo", BackendNativeHTTP, Search),
	p("exa-ai", BackendNativeHTTP, Search),
	p("firecrawl", BackendNativeHTTP, Search),
	p("google-pse", BackendNativeHTTP, Search),
	p("linkup", BackendNativeHTTP, Search),
	p("parallel-ai", BackendNativeHTTP, Search),
	p("searxng", BackendNativeHTTP, Search),
	p("serper", BackendNativeHTTP, Search),
	p("tavily", BackendNativeHTTP, Search),
	p("tinyfish", BackendNativeHTTP, Search),
	p("you-com", BackendNativeHTTP, Search),
}

var explicitByID = func() map[string]Provider {
	result := make(map[string]Provider, len(explicit))
	for _, provider := range explicit {
		result[provider.ID] = provider
	}
	return result
}()
