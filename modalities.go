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
