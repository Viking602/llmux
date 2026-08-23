package llmux

import (
	"context"
	"encoding/json"
)

type ModelCallOptions struct {
	Headers         map[string]string          `json:"headers,omitempty"`
	ProviderOptions map[string]json.RawMessage `json:"providerOptions,omitempty"`
	BodyOverrides   json.RawMessage            `json:"bodyOverrides,omitempty"`
}

type EmbeddingOptions struct {
	ModelCallOptions
	Dimensions *int   `json:"dimensions,omitempty"`
	InputType  string `json:"inputType,omitempty"`
	Truncate   string `json:"truncate,omitempty"`
}

type EmbeddingResult struct {
	Embeddings       [][]float32      `json:"embeddings"`
	InputTokens      int              `json:"inputTokens,omitempty"`
	Response         ResponseMetadata `json:"response,omitempty"`
	ProviderMetadata json.RawMessage  `json:"providerMetadata,omitempty"`
	Warnings         []string         `json:"warnings,omitempty"`
}

type EmbeddingModel interface {
	ModelID() string
	Embed(context.Context, []string, EmbeddingOptions) (EmbeddingResult, error)
}

type RerankDocument struct {
	Text string          `json:"text,omitempty"`
	Data json.RawMessage `json:"data,omitempty"`
}

type RerankOptions struct {
	ModelCallOptions
	TopN *int `json:"topN,omitempty"`
}

type RerankItem struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevanceScore"`
}

type RerankResult struct {
	Ranking          []RerankItem     `json:"ranking"`
	Response         ResponseMetadata `json:"response,omitempty"`
	ProviderMetadata json.RawMessage  `json:"providerMetadata,omitempty"`
	Warnings         []string         `json:"warnings,omitempty"`
}

type RerankingModel interface {
	ModelID() string
	Rerank(context.Context, string, []RerankDocument, RerankOptions) (RerankResult, error)
}

type SpeechRequest struct {
	ModelCallOptions
	Text         string   `json:"text"`
	Voice        string   `json:"voice,omitempty"`
	OutputFormat string   `json:"outputFormat,omitempty"`
	Instructions string   `json:"instructions,omitempty"`
	Speed        *float64 `json:"speed,omitempty"`
	Language     string   `json:"language,omitempty"`
}

type SpeechResult struct {
	Audio            []byte           `json:"audio"`
	MediaType        string           `json:"mediaType,omitempty"`
	Response         ResponseMetadata `json:"response,omitempty"`
	ProviderMetadata json.RawMessage  `json:"providerMetadata,omitempty"`
	Warnings         []string         `json:"warnings,omitempty"`
}

type SpeechModel interface {
	ModelID() string
	GenerateSpeech(context.Context, SpeechRequest) (SpeechResult, error)
}

type TranscriptionRequest struct {
	ModelCallOptions
	Audio       []byte   `json:"audio"`
	MediaType   string   `json:"mediaType"`
	Filename    string   `json:"filename,omitempty"`
	Language    string   `json:"language,omitempty"`
	Prompt      string   `json:"prompt,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	Format      string   `json:"format,omitempty"`
}

type TranscriptionSegment struct {
	Text  string  `json:"text"`
	Start float64 `json:"start,omitempty"`
	End   float64 `json:"end,omitempty"`
}

type TranscriptionResult struct {
	Text             string                 `json:"text"`
	Segments         []TranscriptionSegment `json:"segments,omitempty"`
	Language         string                 `json:"language,omitempty"`
	DurationSeconds  float64                `json:"durationSeconds,omitempty"`
	Response         ResponseMetadata       `json:"response,omitempty"`
	ProviderMetadata json.RawMessage        `json:"providerMetadata,omitempty"`
	Warnings         []string               `json:"warnings,omitempty"`
}

type TranscriptionModel interface {
	ModelID() string
	Transcribe(context.Context, TranscriptionRequest) (TranscriptionResult, error)
}

type BinaryFile struct {
	Data      []byte `json:"data"`
	MediaType string `json:"mediaType,omitempty"`
	Filename  string `json:"filename,omitempty"`
}

type ImageRequest struct {
	ModelCallOptions
	Prompt      string       `json:"prompt"`
	N           *int         `json:"n,omitempty"`
	Size        string       `json:"size,omitempty"`
	AspectRatio string       `json:"aspectRatio,omitempty"`
	Seed        *int64       `json:"seed,omitempty"`
	Files       []BinaryFile `json:"files,omitempty"`
	Mask        *BinaryFile  `json:"mask,omitempty"`
}

type ImageData struct {
	Data          []byte `json:"data,omitempty"`
	URL           string `json:"url,omitempty"`
	MediaType     string `json:"mediaType,omitempty"`
	RevisedPrompt string `json:"revisedPrompt,omitempty"`
}

type ImageResult struct {
	Images           []ImageData      `json:"images"`
	Response         ResponseMetadata `json:"response,omitempty"`
	ProviderMetadata json.RawMessage  `json:"providerMetadata,omitempty"`
	Warnings         []string         `json:"warnings,omitempty"`
}

type ImageModel interface {
	ModelID() string
	GenerateImage(context.Context, ImageRequest) (ImageResult, error)
}

type VideoRequest struct {
	ModelCallOptions
	Prompt      string `json:"prompt"`
	N           *int   `json:"n,omitempty"`
	AspectRatio string `json:"aspectRatio,omitempty"`
	Resolution  string `json:"resolution,omitempty"`
	Seed        *int64 `json:"seed,omitempty"`
}

type VideoData struct {
	Data      []byte `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	MediaType string `json:"mediaType,omitempty"`
}

type VideoResult struct {
	Videos           []VideoData      `json:"videos"`
	Response         ResponseMetadata `json:"response,omitempty"`
	ProviderMetadata json.RawMessage  `json:"providerMetadata,omitempty"`
	Warnings         []string         `json:"warnings,omitempty"`
}

type VideoModel interface {
	ModelID() string
	GenerateVideo(context.Context, VideoRequest) (VideoResult, error)
}

type UploadRequest struct {
	ModelCallOptions
	BinaryFile
	Purpose string `json:"purpose,omitempty"`
}

type UploadResult struct {
	ProviderReference map[string]string `json:"providerReference"`
	MediaType         string            `json:"mediaType,omitempty"`
	Filename          string            `json:"filename,omitempty"`
	ProviderMetadata  json.RawMessage   `json:"providerMetadata,omitempty"`
	Warnings          []string          `json:"warnings,omitempty"`
}

type Files interface {
	Upload(context.Context, UploadRequest) (UploadResult, error)
}

type SearchRequest struct {
	ModelCallOptions
	Query             string   `json:"query"`
	MaxResults        *int     `json:"maxResults,omitempty"`
	IncludeRawContent *bool    `json:"includeRawContent,omitempty"`
	TimeRange         string   `json:"timeRange,omitempty"`
	IncludeDomains    []string `json:"includeDomains,omitempty"`
	ExcludeDomains    []string `json:"excludeDomains,omitempty"`
}

type SearchItem struct {
	Title      string  `json:"title,omitempty"`
	URL        string  `json:"url,omitempty"`
	Content    string  `json:"content,omitempty"`
	RawContent string  `json:"rawContent,omitempty"`
	Score      float64 `json:"score,omitempty"`
}

type SearchResult struct {
	Results          []SearchItem     `json:"results"`
	Answer           string           `json:"answer,omitempty"`
	Response         ResponseMetadata `json:"response,omitempty"`
	ProviderMetadata json.RawMessage  `json:"providerMetadata,omitempty"`
	Warnings         []string         `json:"warnings,omitempty"`
}

type SearchModel interface {
	Search(context.Context, SearchRequest) (SearchResult, error)
}

// Optional provider factories expose non-language modalities without bloating
// the base Provider interface.
type EmbeddingProvider interface {
	EmbeddingModel(modelID string) (EmbeddingModel, error)
}

type RerankingProvider interface {
	RerankingModel(modelID string) (RerankingModel, error)
}

type SpeechProvider interface {
	SpeechModel(modelID string) (SpeechModel, error)
}

type TranscriptionProvider interface {
	TranscriptionModel(modelID string) (TranscriptionModel, error)
}

type ImageProvider interface {
	ImageModel(modelID string) (ImageModel, error)
}

type VideoProvider interface {
	VideoModel(modelID string) (VideoModel, error)
}

type SearchProvider interface {
	SearchModel(modelID string) (SearchModel, error)
}

type FilesProvider interface {
	Files() Files
}

func OpenEmbeddingModel(provider Provider, modelID string) (EmbeddingModel, error) {
	factory, ok := provider.(EmbeddingProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "embedding models")
	}
	return factory.EmbeddingModel(modelID)
}

func OpenRerankingModel(provider Provider, modelID string) (RerankingModel, error) {
	factory, ok := provider.(RerankingProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "reranking models")
	}
	return factory.RerankingModel(modelID)
}

func OpenSpeechModel(provider Provider, modelID string) (SpeechModel, error) {
	factory, ok := provider.(SpeechProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "speech models")
	}
	return factory.SpeechModel(modelID)
}

func OpenTranscriptionModel(provider Provider, modelID string) (TranscriptionModel, error) {
	factory, ok := provider.(TranscriptionProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "transcription models")
	}
	return factory.TranscriptionModel(modelID)
}

func OpenImageModel(provider Provider, modelID string) (ImageModel, error) {
	factory, ok := provider.(ImageProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "image models")
	}
	return factory.ImageModel(modelID)
}

func OpenVideoModel(provider Provider, modelID string) (VideoModel, error) {
	factory, ok := provider.(VideoProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "video models")
	}
	return factory.VideoModel(modelID)
}

func OpenSearchModel(provider Provider, modelID string) (SearchModel, error) {
	factory, ok := provider.(SearchProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "search models")
	}
	return factory.SearchModel(modelID)
}

func OpenFiles(provider Provider) (Files, error) {
	factory, ok := provider.(FilesProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "files")
	}
	return factory.Files(), nil
}

func unsupportedProviderCapability(provider Provider, capability string) error {
	if provider == nil {
		return &ProviderError{Kind: ErrorInvalidRequest, Message: "provider is nil"}
	}
	return &ProviderError{
		Provider: provider.Name(),
		Kind:     ErrorUnsupported,
		Message:  capability + " are not supported by this provider",
	}
}
