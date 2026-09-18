// Package vercel implements AI Gateway's typed evaluation protocol.
package vercel

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

const DefaultBaseURL = "https://ai-gateway.vercel.sh/v4/ai"

type Config struct {
	APIKey  string
	BaseURL string
	Client  *http.Client
}

type Provider struct{ config Config }

func New(config Config) (*Provider, error) {
	if strings.TrimSpace(config.APIKey) == "" {
		return nil, errors.New("vercel: API key required")
	}
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	u, err := url.Parse(config.BaseURL)
	if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("vercel: invalid base URL")
	}
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	return &Provider{config: config}, nil
}

func (*Provider) Name() string { return "vercel-evaluation" }

func (provider *Provider) LanguageModel(string) (llmux.LanguageModel, error) {
	return nil, &llmux.ProviderError{Provider: provider.Name(), Kind: llmux.ErrorUnsupported, Code: "EVALUATION_ONLY"}
}

type evaluationModel struct {
	provider *Provider
	id       string
}

func (provider *Provider) EvaluationModel(id string) (llmux.EvaluationModel, error) {
	if strings.TrimSpace(id) == "" {
		return nil, errors.New("vercel: model ID required")
	}
	return &evaluationModel{provider: provider, id: id}, nil
}

func (model *evaluationModel) ModelID() string { return model.id }

func (model *evaluationModel) Evaluate(ctx context.Context, input llmux.EvaluationRequest) (llmux.EvaluationResult, error) {
	var result llmux.EvaluationResult
	if err := input.Validate(); err != nil {
		return result, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return result, err
	}
	config := model.provider.config
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+config.APIKey)
	headers.Set("Content-Type", "application/json")
	headers.Set("ai-gateway-protocol-version", "0.0.1")
	headers.Set("ai-gateway-auth-method", "api-key")
	headers.Set("ai-evaluation-model-specification-version", "4")
	headers.Set("ai-model-id", model.id)
	// Evaluation may be billed even when a response is lost. The caller owns recovery.
	response, err := httpx.Do(ctx, config.Client, httpx.Request{Method: http.MethodPost, URL: config.BaseURL + "/evaluation-model", Headers: headers, Body: body, Retry: llmux.RetryPolicy{MaxAttempts: 1}})
	if err != nil {
		return result, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		upstream := &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorKindForStatus(response.StatusCode), StatusCode: response.StatusCode, Code: "EVALUATION_REJECTED"}
		var body struct {
			Error struct {
				Message string `json:"message"`
				Code    string `json:"code"`
			} `json:"error"`
			Message string `json:"message"`
		}
		// Only explicit diagnostic fields; no arbitrary body or HTML fallback.
		if json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&body) == nil {
			upstream.Message = body.Error.Message
			if upstream.Message == "" {
				upstream.Message = body.Message
			}
			if body.Error.Code != "" {
				upstream.Code = body.Error.Code
			}
			if len(upstream.Message) > 8192 {
				upstream.Message = upstream.Message[:8192]
			}
		}
		if seconds, err := strconv.Atoi(response.Header.Get("Retry-After")); err == nil && seconds > 0 {
			upstream.RetryAfter = time.Duration(min(seconds, 86400)) * time.Second
		} else if at, err := http.ParseTime(response.Header.Get("Retry-After")); err == nil {
			upstream.RetryAfter = max(0, min(time.Until(at), 24*time.Hour))
		}
		return result, upstream
	}
	const maxResponseBytes = 4 << 20
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return result, err
	}
	if len(payload) > maxResponseBytes || json.Unmarshal(payload, &result) != nil || input.ValidateAnswers(result) != nil {
		return llmux.EvaluationResult{}, &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorUnknown, Code: "EVALUATION_RESPONSE_INVALID"}
	}
	result.Response.ModelID = model.id
	result.Response.ID = response.Header.Get("x-request-id")
	return result, nil
}
