// Package typesafe implements TypeSafe AI's System One API for Jev.
package typesafe

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

const (
	APIKeyEnvVar     = "TYPESAFE_API_KEY"
	DefaultModel     = "jev-latest"
	DefaultBaseURL   = "https://api.typesafe.ai/v1"
	maxResponseBytes = 16 << 20
)

type Config struct {
	APIKey  string
	BaseURL string // Includes /v1; defaults to DefaultBaseURL.
	Headers http.Header
	Client  *http.Client
	Retry   llmux.RetryPolicy
}

type Provider struct{ config Config }
type model struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if strings.TrimSpace(config.APIKey) == "" || strings.ContainsAny(config.APIKey, "\r\n") {
		return nil, providerError(llmux.ErrorInvalidRequest, "API key is empty or invalid")
	}
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, providerError(llmux.ErrorInvalidRequest, "invalid base URL")
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	config.Headers = config.Headers.Clone()
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	return &Provider{config: config}, nil
}

func (*Provider) Name() string { return "typesafe" }

func (*Provider) Descriptor() llmux.ProviderDescriptor {
	return llmux.ProviderDescriptor{
		Name: "typesafe", WireProtocols: []string{"typesafe-system-one"},
		Authentication: []string{"bearer"},
		Capabilities:   []llmux.ProviderCapability{llmux.CapabilityEvaluation, llmux.CapabilityModelListing},
	}
}

func (*Provider) LanguageModel(string) (llmux.LanguageModel, error) {
	return nil, providerError(llmux.ErrorUnsupported, "Jev evaluates typed questions; use EvaluationModel")
}

func (provider *Provider) EvaluationModel(modelID string) (llmux.EvaluationModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, providerError(llmux.ErrorInvalidRequest, "model ID is empty")
	}
	return &model{provider: provider, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (model *model) Evaluate(ctx context.Context, request llmux.EvaluationRequest) (llmux.EvaluationResult, error) {
	payload, questions, err := buildRequest(model.id, request)
	if err != nil {
		return llmux.EvaluationResult{}, providerError(llmux.ErrorInvalidRequest, err.Error())
	}
	payload, err = model.provider.do(ctx, http.MethodPost, "/systemone", payload, request.Headers)
	if err != nil {
		return llmux.EvaluationResult{}, err
	}
	result, err := parseResult(payload, questions)
	if err != nil {
		return llmux.EvaluationResult{}, providerError(llmux.ErrorUnknown, "invalid evaluation response: "+err.Error())
	}
	return result, nil
}

func (provider *Provider) ListModels(ctx context.Context) ([]llmux.ModelInfo, error) {
	payload, err := provider.do(ctx, http.MethodGet, "/models", nil, nil)
	if err != nil {
		return nil, err
	}
	var wire struct {
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil || wire.Models == nil {
		return nil, providerError(llmux.ErrorUnknown, "invalid models response")
	}
	result := make([]llmux.ModelInfo, 0, len(wire.Models))
	for _, raw := range wire.Models {
		var entry struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			ReleaseDate string `json:"release_date"`
		}
		if err := json.Unmarshal(raw, &entry); err != nil || strings.TrimSpace(entry.Name) == "" {
			return nil, providerError(llmux.ErrorUnknown, "invalid model entry")
		}
		var created int64
		if date, err := time.Parse(time.DateOnly, entry.ReleaseDate); err == nil {
			created = date.Unix()
		}
		yes, no := true, false
		result = append(result, llmux.ModelInfo{
			ID: entry.Name, DisplayName: entry.Name, Description: entry.Description,
			OwnedBy: provider.Name(), Created: created, Raw: raw,
			Capabilities: &llmux.ModelCapabilities{
				InputModalities:  []llmux.Modality{llmux.ModalityText},
				OutputModalities: []llmux.Modality{llmux.ModalityEvaluation},
				StructuredOutput: &yes, Streaming: &no, ToolCalling: &no,
			},
		})
	}
	return result, nil
}

func (provider *Provider) do(ctx context.Context, method, path string, body []byte, overrides map[string]string) ([]byte, error) {
	headers := provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	for key, value := range overrides {
		headers.Set(key, value)
	}
	headers.Set("Authorization", "Bearer "+provider.config.APIKey)
	headers.Set("Accept", "application/json")
	if body != nil {
		headers.Set("Content-Type", "application/json")
	}
	response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{
		Method: method, URL: provider.config.BaseURL + path, Headers: headers,
		Body: body, Retry: provider.config.Retry,
	})
	if err != nil {
		return nil, transportError(err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return nil, transportError(err)
	}
	if len(payload) > maxResponseBytes {
		return nil, providerError(llmux.ErrorUnknown, "response exceeds 16 MiB")
	}
	if response.StatusCode/100 != 2 {
		return nil, responseError(response, payload)
	}
	return payload, nil
}

func responseError(response *http.Response, payload []byte) error {
	var envelope struct {
		Detail json.RawMessage `json:"detail"`
	}
	_ = json.Unmarshal(payload, &envelope)
	var detail struct {
		Type    string `json:"error_type"`
		Message string `json:"message"`
	}
	_ = json.Unmarshal(envelope.Detail, &detail)
	message := detail.Message
	if message == "" {
		_ = json.Unmarshal(envelope.Detail, &message)
	}
	if message == "" {
		message = strings.TrimSpace(string(payload))
	}
	err := &llmux.ProviderError{
		Provider: "typesafe", Kind: llmux.ErrorKindForStatus(response.StatusCode),
		StatusCode: response.StatusCode, Code: detail.Type,
		Message: message[:min(len(message), 8<<10)],
	}
	// The live API also returns 403 for a missing key, with this error type.
	if detail.Type == "authentication_error" && (response.StatusCode == 401 || response.StatusCode == 403) {
		err.Kind = llmux.ErrorAuthentication
	}
	if json.Valid(payload) {
		err.Raw = payload
	}
	value := response.Header.Get("Retry-After")
	if seconds, parseErr := strconv.ParseFloat(value, 64); parseErr == nil && seconds > 0 && seconds < float64((1<<63-1)/int64(time.Second)) {
		err.RetryAfter = time.Duration(seconds * float64(time.Second))
	} else if date, parseErr := http.ParseTime(value); parseErr == nil {
		err.RetryAfter = max(0, time.Until(date))
	}
	return err
}

func providerError(kind llmux.ErrorKind, message string) *llmux.ProviderError {
	return &llmux.ProviderError{Provider: "typesafe", Kind: kind, Message: message}
}

func transportError(err error) error {
	kind := llmux.ErrorStream
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		kind = llmux.ErrorCancelled
	}
	return &llmux.ProviderError{Provider: "typesafe", Kind: kind, Message: err.Error(), Cause: err}
}

var (
	_ llmux.Provider           = (*Provider)(nil)
	_ llmux.EvaluationProvider = (*Provider)(nil)
	_ llmux.ModelLister        = (*Provider)(nil)
)
