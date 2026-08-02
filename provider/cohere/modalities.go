package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

type embeddingModel struct {
	provider *Provider
	id       string
}
type rerankingModel struct {
	provider *Provider
	id       string
}

func (provider *Provider) EmbeddingModel(modelID string) (llmux.EmbeddingModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("cohere: model ID is empty")
	}
	return &embeddingModel{provider: provider, id: modelID}, nil
}
func (provider *Provider) RerankingModel(modelID string) (llmux.RerankingModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("cohere: model ID is empty")
	}
	return &rerankingModel{provider: provider, id: modelID}, nil
}
func (model *embeddingModel) ModelID() string { return model.id }
func (model *rerankingModel) ModelID() string { return model.id }

func (model *embeddingModel) Embed(ctx context.Context, values []string, options llmux.EmbeddingOptions) (llmux.EmbeddingResult, error) {
	if len(values) == 0 {
		return llmux.EmbeddingResult{}, errors.New("cohere: embedding values are empty")
	}
	body := map[string]any{"model": model.id, "texts": values, "embedding_types": []string{"float"}, "input_type": first(options.InputType, "search_query")}
	if options.Truncate != "" {
		body["truncate"] = options.Truncate
	}
	if options.Dimensions != nil {
		body["output_dimension"] = *options.Dimensions
	}
	if err := mergeOptions(body, options.ModelCallOptions, "model", "texts"); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	response, err := model.do(ctx, "/embed", body, options.Headers)
	if err != nil {
		return llmux.EmbeddingResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Embeddings struct {
			Float [][]float32 `json:"float"`
		} `json:"embeddings"`
		Meta struct {
			BilledUnits struct {
				InputTokens int `json:"input_tokens"`
			} `json:"billed_units"`
		} `json:"meta"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64<<20)).Decode(&wire); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	return llmux.EmbeddingResult{Embeddings: wire.Embeddings.Float, InputTokens: wire.Meta.BilledUnits.InputTokens, Response: llmux.ResponseMetadata{ModelID: model.id}}, nil
}

func (model *rerankingModel) Rerank(ctx context.Context, query string, documents []llmux.RerankDocument, options llmux.RerankOptions) (llmux.RerankResult, error) {
	if query == "" || len(documents) == 0 {
		return llmux.RerankResult{}, errors.New("cohere: rerank query and documents are required")
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
	body := map[string]any{"model": model.id, "query": query, "documents": values}
	if options.TopN != nil {
		body["top_n"] = *options.TopN
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
		ID      string `json:"id"`
		Results []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 16<<20)).Decode(&wire); err != nil {
		return llmux.RerankResult{}, err
	}
	result := llmux.RerankResult{Ranking: make([]llmux.RerankItem, len(wire.Results)), Response: llmux.ResponseMetadata{ID: wire.ID, ModelID: model.id}, Warnings: warnings}
	for index, item := range wire.Results {
		result.Ranking[index] = llmux.RerankItem{Index: item.Index, RelevanceScore: item.Score}
	}
	return result, nil
}

func (model *embeddingModel) do(ctx context.Context, path string, body map[string]any, overrides map[string]string) (*http.Response, error) {
	return doModality(ctx, model.provider, path, body, overrides)
}
func (model *rerankingModel) do(ctx context.Context, path string, body map[string]any, overrides map[string]string) (*http.Response, error) {
	return doModality(ctx, model.provider, path, body, overrides)
}

func doModality(ctx context.Context, provider *Provider, path string, body map[string]any, overrides map[string]string) (*http.Response, error) {
	payload, _ := json.Marshal(body)
	headers := (&model{provider: provider}).headers(overrides)
	response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{Method: http.MethodPost, URL: provider.config.BaseURL + path, Headers: headers, Body: payload, Retry: provider.config.Retry})
	if err != nil {
		return nil, (&model{provider: provider}).transportError(err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		return nil, (&model{provider: provider}).responseError(response)
	}
	return response, nil
}

func mergeOptions(body map[string]any, options llmux.ModelCallOptions, protected ...string) error {
	blocked := make(map[string]bool)
	for _, key := range protected {
		blocked[key] = true
	}
	for _, raw := range []json.RawMessage{options.ProviderOptions["cohere"], options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if json.Unmarshal(raw, &extra) != nil || extra == nil {
			return errors.New("cohere: modality options must be JSON objects")
		}
		for key, value := range extra {
			if blocked[key] {
				return errors.New("cohere: protected modality field cannot be overridden: " + key)
			}
			switch key {
			case "inputType":
				key = "input_type"
			case "outputDimension":
				key = "output_dimension"
			case "maxTokensPerDoc":
				key = "max_tokens_per_doc"
			}
			body[key] = value
		}
	}
	return nil
}
