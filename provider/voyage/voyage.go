package voyage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

type Config struct {
	APIKey  string
	BaseURL string
	Headers http.Header
	Client  *http.Client
	Retry   llmux.RetryPolicy
}
type Provider struct{ config Config }
type embeddingModel struct {
	provider *Provider
	id       string
}
type rerankingModel struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if config.APIKey == "" {
		return nil, errors.New("voyage: API key is empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://api.voyageai.com/v1"
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("voyage: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	config.Headers = config.Headers.Clone()
	return &Provider{config: config}, nil
}

func (provider *Provider) Name() string { return "voyage" }
func (provider *Provider) EmbeddingModel(modelID string) (llmux.EmbeddingModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("voyage: model ID is empty")
	}
	return &embeddingModel{provider: provider, id: modelID}, nil
}
func (provider *Provider) RerankingModel(modelID string) (llmux.RerankingModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("voyage: model ID is empty")
	}
	return &rerankingModel{provider: provider, id: modelID}, nil
}
func (model *embeddingModel) ModelID() string { return model.id }
func (model *rerankingModel) ModelID() string { return model.id }

func (model *embeddingModel) Embed(ctx context.Context, values []string, options llmux.EmbeddingOptions) (llmux.EmbeddingResult, error) {
	if len(values) == 0 {
		return llmux.EmbeddingResult{}, errors.New("voyage: embedding values are empty")
	}
	body := map[string]any{"input": values, "model": model.id}
	if options.InputType != "" {
		body["input_type"] = options.InputType
	}
	if options.Truncate != "" {
		body["truncation"] = options.Truncate != "NONE"
	}
	if options.Dimensions != nil {
		body["output_dimension"] = *options.Dimensions
	}
	if err := mergeOptions(body, options.ModelCallOptions, "model", "input"); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	response, err := model.do(ctx, "/embeddings", body, options.Headers)
	if err != nil {
		return llmux.EmbeddingResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64<<20)).Decode(&wire); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	sort.Slice(wire.Data, func(i, j int) bool { return wire.Data[i].Index < wire.Data[j].Index })
	result := llmux.EmbeddingResult{Embeddings: make([][]float32, len(wire.Data)), InputTokens: wire.Usage.TotalTokens, Response: llmux.ResponseMetadata{ModelID: model.id}}
	for index := range wire.Data {
		result.Embeddings[index] = wire.Data[index].Embedding
	}
	return result, nil
}

func (model *rerankingModel) Rerank(ctx context.Context, query string, documents []llmux.RerankDocument, options llmux.RerankOptions) (llmux.RerankResult, error) {
	if query == "" || len(documents) == 0 {
		return llmux.RerankResult{}, errors.New("voyage: rerank query and documents are required")
	}
	values := make([]string, len(documents))
	warnings := make([]string, 0)
	for index, document := range documents {
		if document.Text != "" {
			values[index] = document.Text
		} else {
			values[index] = string(document.Data)
			warnings = append(warnings, "object document was converted to JSON text")
		}
	}
	body := map[string]any{"query": query, "documents": values, "model": model.id}
	if options.TopN != nil {
		body["top_k"] = *options.TopN
	}
	if err := mergeOptions(body, options.ModelCallOptions, "model", "query", "documents"); err != nil {
		return llmux.RerankResult{}, err
	}
	response, err := model.do(ctx, "/rerank", body, options.Headers)
	if err != nil {
		return llmux.RerankResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Data []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 16<<20)).Decode(&wire); err != nil {
		return llmux.RerankResult{}, err
	}
	result := llmux.RerankResult{Ranking: make([]llmux.RerankItem, len(wire.Data)), Response: llmux.ResponseMetadata{ModelID: model.id}, Warnings: warnings}
	for index, item := range wire.Data {
		result.Ranking[index] = llmux.RerankItem{Index: item.Index, RelevanceScore: item.Score}
	}
	return result, nil
}

func (model *embeddingModel) do(ctx context.Context, path string, body map[string]any, overrides map[string]string) (*http.Response, error) {
	return do(ctx, model.provider, path, body, overrides)
}
func (model *rerankingModel) do(ctx context.Context, path string, body map[string]any, overrides map[string]string) (*http.Response, error) {
	return do(ctx, model.provider, path, body, overrides)
}
func do(ctx context.Context, provider *Provider, path string, body map[string]any, overrides map[string]string) (*http.Response, error) {
	payload, _ := json.Marshal(body)
	headers := provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Authorization", "Bearer "+provider.config.APIKey)
	headers.Set("Content-Type", "application/json")
	for key, value := range overrides {
		headers.Set(key, value)
	}
	response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{Method: http.MethodPost, URL: provider.config.BaseURL + path, Headers: headers, Body: payload, Retry: provider.config.Retry})
	if err != nil {
		return nil, &llmux.ProviderError{Provider: "voyage", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	if response.StatusCode/100 != 2 {
		payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
		_ = response.Body.Close()
		var envelope struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(payload, &envelope)
		return nil, &llmux.ProviderError{Provider: "voyage", Kind: llmux.ErrorKindForStatus(response.StatusCode), StatusCode: response.StatusCode, Message: first(envelope.Detail, string(payload))}
	}
	return response, nil
}
func mergeOptions(body map[string]any, options llmux.ModelCallOptions, protected ...string) error {
	blocked := make(map[string]bool)
	for _, key := range protected {
		blocked[key] = true
	}
	for _, raw := range []json.RawMessage{options.ProviderOptions["voyage"], options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if json.Unmarshal(raw, &extra) != nil || extra == nil {
			return errors.New("voyage: modality options must be JSON objects")
		}
		for key, value := range extra {
			if blocked[key] {
				return errors.New("voyage: protected modality field cannot be overridden: " + key)
			}
			switch key {
			case "inputType":
				key = "input_type"
			case "outputDimension":
				key = "output_dimension"
			case "outputDtype":
				key = "output_dtype"
			case "returnDocuments":
				key = "return_documents"
			}
			body[key] = value
		}
	}
	return nil
}
func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
