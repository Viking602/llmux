package tavily

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
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

type Model struct{ config Config }

func New(config Config) (*Model, error) {
	if config.APIKey == "" {
		return nil, errors.New("tavily: API key is empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://api.tavily.com"
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("tavily: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	config.Headers = config.Headers.Clone()
	return &Model{config: config}, nil
}

func (model *Model) Search(ctx context.Context, request llmux.SearchRequest) (llmux.SearchResult, error) {
	if strings.TrimSpace(request.Query) == "" {
		return llmux.SearchResult{}, errors.New("tavily: query is empty")
	}
	body := map[string]any{"query": request.Query, "include_answer": false}
	if request.MaxResults != nil {
		body["max_results"] = *request.MaxResults
	}
	if request.IncludeRawContent != nil {
		body["include_raw_contents"] = *request.IncludeRawContent
	}
	if request.TimeRange != "" {
		body["time_range"] = request.TimeRange
	}
	if len(request.IncludeDomains) > 0 {
		body["include_domains"] = request.IncludeDomains
	}
	if len(request.ExcludeDomains) > 0 {
		body["exclude_domains"] = request.ExcludeDomains
	}
	for _, raw := range []json.RawMessage{request.ProviderOptions["tavily"], request.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if json.Unmarshal(raw, &extra) != nil || extra == nil {
			return llmux.SearchResult{}, errors.New("tavily: options must be JSON objects")
		}
		for key, value := range extra {
			if key == "query" {
				return llmux.SearchResult{}, errors.New("tavily: query cannot be overridden")
			}
			body[key] = value
		}
	}
	payload, _ := json.Marshal(body)
	headers := model.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Authorization", "Bearer "+model.config.APIKey)
	headers.Set("Content-Type", "application/json")
	for key, value := range request.Headers {
		headers.Set(key, value)
	}
	response, err := httpx.Do(ctx, model.config.Client, httpx.Request{Method: http.MethodPost, URL: model.config.BaseURL + "/search", Headers: headers, Body: payload, Retry: model.config.Retry})
	if err != nil {
		return llmux.SearchResult{}, &llmux.ProviderError{Provider: "tavily", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		payload, _ = io.ReadAll(io.LimitReader(response.Body, 4<<20))
		var envelope struct {
			Detail struct {
				Error string `json:"error"`
			} `json:"detail"`
		}
		_ = json.Unmarshal(payload, &envelope)
		message := envelope.Detail.Error
		if message == "" {
			message = string(payload)
		}
		return llmux.SearchResult{}, &llmux.ProviderError{Provider: "tavily", Kind: llmux.ErrorKindForStatus(response.StatusCode), StatusCode: response.StatusCode, Message: message}
	}
	var wire struct {
		Answer  string `json:"answer"`
		Results []struct {
			Title      string  `json:"title"`
			URL        string  `json:"url"`
			Content    string  `json:"content"`
			RawContent string  `json:"raw_content"`
			Score      float64 `json:"score"`
		} `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&wire); err != nil {
		return llmux.SearchResult{}, fmt.Errorf("tavily: decode response: %w", err)
	}
	result := llmux.SearchResult{Answer: wire.Answer, Results: make([]llmux.SearchItem, len(wire.Results))}
	for index, item := range wire.Results {
		result.Results[index] = llmux.SearchItem{Title: item.Title, URL: item.URL, Content: item.Content, RawContent: item.RawContent, Score: item.Score}
	}
	return result, nil
}
