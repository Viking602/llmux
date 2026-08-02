package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

const defaultBaseURL = "https://api.openai.com/v1"

type WireAPI string

const (
	ChatCompletions WireAPI = "chat_completions"
	Responses       WireAPI = "responses"
)

type CompatProfile struct {
	SupportsTopK            bool
	SupportsTools           bool
	SupportsResponseFormat  bool
	StreamUsageKey          string
	DeepSeek                bool
	RequiredToolChoice      string
	SupportsStop            bool
	SupportsPenalties       bool
	UsesMaxCompletionTokens bool
	XAI                     bool
}

func FullProfile() CompatProfile {
	return CompatProfile{SupportsTopK: true, SupportsTools: true, SupportsResponseFormat: true, RequiredToolChoice: "required", SupportsStop: true, SupportsPenalties: true}
}

type Config struct {
	APIKey           string
	BaseURL          string
	Organization     string
	Project          string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	WireAPI          WireAPI
	ProviderName     string
	AllowEmptyAPIKey bool
	Profile          *CompatProfile
	APIKeyHeader     string
	APIKeyPrefix     string
	Endpoint         string
}

type Provider struct {
	config Config
}

type model struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if strings.TrimSpace(config.APIKey) == "" && !config.AllowEmptyAPIKey {
		return nil, errors.New("openai: API key is empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = defaultBaseURL
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("openai: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.Endpoint != "" {
		endpoint, endpointErr := url.Parse(config.Endpoint)
		if endpointErr != nil || endpoint.Scheme == "" || endpoint.Host == "" {
			return nil, fmt.Errorf("openai: invalid endpoint %q", config.Endpoint)
		}
	}
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	if config.WireAPI == "" {
		config.WireAPI = Responses
	}
	if config.WireAPI != ChatCompletions && config.WireAPI != Responses {
		return nil, fmt.Errorf("openai: unsupported wire API %q", config.WireAPI)
	}
	if config.ProviderName == "" {
		config.ProviderName = "openai"
	}
	if config.APIKeyHeader == "" {
		config.APIKeyHeader = "Authorization"
		if config.APIKeyPrefix == "" {
			config.APIKeyPrefix = "Bearer "
		}
	}
	config.Headers = config.Headers.Clone()
	if config.Profile == nil {
		profile := FullProfile()
		config.Profile = &profile
	} else {
		profile := *config.Profile
		config.Profile = &profile
	}
	return &Provider{config: config}, nil
}

func (provider *Provider) Name() string { return provider.config.ProviderName }

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("openai: model ID is empty")
	}
	return &model{provider: provider, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	body, err := model.build(request, false)
	if err != nil {
		return llmux.Result{}, err
	}
	retry := model.provider.config.Retry
	if request.Options.MaxRetries != nil {
		retry.MaxAttempts = max(*request.Options.MaxRetries, 0) + 1
	}
	response, err := httpx.Do(ctx, model.provider.config.Client, httpx.Request{
		Method: http.MethodPost, URL: model.endpoint(), Headers: model.headers(request.Options.Headers), Body: body, Retry: retry,
	})
	if err != nil {
		return llmux.Result{}, model.transportError(err)
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return llmux.Result{}, model.responseError(response)
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return llmux.Result{}, model.transportError(err)
	}
	var result llmux.Result
	if model.provider.config.WireAPI == Responses {
		result, err = parseResponsesResult(payload)
	} else {
		result, err = parseChatResult(payload)
	}
	if err != nil {
		return llmux.Result{}, &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	result.Response.Headers = selectedHeaders(response.Header)
	if model.provider.config.Profile.XAI {
		if result.Usage.CachedInputTokens > result.Usage.InputTokens {
			result.Usage.InputTokens += result.Usage.CachedInputTokens
		}
		result.Usage.OutputTokens += result.Usage.ReasoningTokens
		result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.OutputTokens
	}
	result.Warnings = append(result.Warnings, model.profileWarnings(request.Options)...)
	if request.Options.IncludeRawChunks {
		result.Raw = append(json.RawMessage(nil), payload...)
	}
	return result, nil
}

func (model *model) Stream(ctx context.Context, request llmux.Request) (llmux.Stream, error) {
	body, err := model.build(request, true)
	if err != nil {
		return nil, err
	}
	streamCtx, cancel := context.WithCancel(ctx)
	retry := model.provider.config.Retry
	if request.Options.MaxRetries != nil {
		retry.MaxAttempts = max(*request.Options.MaxRetries, 0) + 1
	}
	response, err := httpx.Do(streamCtx, model.provider.config.Client, httpx.Request{
		Method: http.MethodPost, URL: model.endpoint(), Headers: model.headers(request.Options.Headers), Body: body, Retry: retry,
	})
	if err != nil {
		cancel()
		return nil, model.transportError(err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		cancel()
		return nil, model.responseError(response)
	}
	contentType := response.Header.Get("Content-Type")
	if contentType != "" {
		mediaType, _, parseErr := mime.ParseMediaType(contentType)
		if parseErr != nil || mediaType != "text/event-stream" {
			payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
			_ = response.Body.Close()
			cancel()
			var envelope struct {
				Code  string          `json:"code"`
				Error json.RawMessage `json:"error"`
			}
			if json.Unmarshal(payload, &envelope) == nil && len(envelope.Error) > 0 {
				var message string
				if json.Unmarshal(envelope.Error, &message) != nil {
					message = string(envelope.Error)
				}
				return nil, &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorStream, Code: envelope.Code, Message: message, Raw: payload}
			}
			return nil, fmt.Errorf("openai: expected text/event-stream, got %q", contentType)
		}
	}
	metadata := llmux.ResponseMetadata{ModelID: model.id, Headers: selectedHeaders(response.Header)}
	if model.provider.config.WireAPI == Responses {
		return newResponsesStream(streamCtx, cancel, response.Body, model.provider.Name(), metadata, request.Options.IncludeRawChunks, model.profileWarnings(request.Options)), nil
	}
	return newChatStream(streamCtx, cancel, response.Body, model.provider.Name(), metadata, request.Options.IncludeRawChunks, *model.provider.config.Profile, model.profileWarnings(request.Options)), nil
}

func (model *model) profileWarnings(options llmux.CallOptions) []string {
	profile := model.provider.config.Profile
	warnings := make([]string, 0, 3)
	if options.TopK != nil && !profile.SupportsTopK {
		warnings = append(warnings, "topK is not supported by this provider and was omitted")
	}
	if len(options.Tools) > 0 && !profile.SupportsTools {
		warnings = append(warnings, "tools are not supported by this provider and were omitted")
	}
	if options.ResponseFormat != nil && !profile.SupportsResponseFormat {
		warnings = append(warnings, "response format is not supported by this provider and was omitted")
	}
	if len(options.StopSequences) > 0 && !profile.SupportsStop {
		warnings = append(warnings, "stop sequences are not supported by this provider and were omitted")
	}
	if (options.PresencePenalty != nil || options.FrequencyPenalty != nil) && !profile.SupportsPenalties {
		warnings = append(warnings, "presence and frequency penalties are not supported by this provider and were omitted")
	}
	if options.Reasoning != nil && profile.XAI && xaiRejectsReasoningEffort(model.id) {
		warnings = append(warnings, "reasoning effort is not supported by this xAI model and was omitted")
	}
	return warnings
}

func (model *model) endpoint() string {
	if model.provider.config.Endpoint != "" {
		return model.provider.config.Endpoint
	}
	if model.provider.config.WireAPI == Responses {
		return model.provider.config.BaseURL + "/responses"
	}
	return model.provider.config.BaseURL + "/chat/completions"
}

func (model *model) headers(overrides map[string]string) http.Header {
	headers := model.provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Accept", "application/json")
	headers.Set("Content-Type", "application/json")
	if model.provider.config.APIKey != "" {
		headers.Set(model.provider.config.APIKeyHeader, model.provider.config.APIKeyPrefix+model.provider.config.APIKey)
	}
	if model.provider.config.Organization != "" {
		headers.Set("OpenAI-Organization", model.provider.config.Organization)
	}
	if model.provider.config.Project != "" {
		headers.Set("OpenAI-Project", model.provider.config.Project)
	}
	for name, value := range overrides {
		if !forbiddenHeader(name) {
			headers.Set(name, value)
		}
	}
	return headers
}

func forbiddenHeader(name string) bool {
	switch http.CanonicalHeaderKey(name) {
	case "Host", "Content-Length", "Transfer-Encoding":
		return true
	default:
		return false
	}
}

func (model *model) responseError(response *http.Response) error {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
		Message string `json:"message"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := first(envelope.Error.Message, envelope.Message, strings.TrimSpace(string(payload)))
	code := envelope.Error.Type
	if envelope.Error.Code != nil {
		code = fmt.Sprint(envelope.Error.Code)
	}
	providerError := &llmux.ProviderError{
		Provider: model.provider.Name(), Kind: llmux.ErrorKindForStatus(response.StatusCode), Code: code,
		StatusCode: response.StatusCode, Message: bounded(message, 8<<10), RetryAfter: retryAfter(response.Header.Get("Retry-After")),
	}
	if json.Valid(payload) {
		providerError.Raw = append(json.RawMessage(nil), payload...)
	}
	return providerError
}

func (model *model) transportError(err error) error {
	kind := llmux.ErrorStream
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		kind = llmux.ErrorCancelled
	}
	return &llmux.ProviderError{Provider: model.provider.Name(), Kind: kind, Message: err.Error(), Cause: err}
}

func selectedHeaders(headers http.Header) map[string]string {
	result := make(map[string]string)
	for _, key := range []string{"Request-Id", "X-Request-Id", "Openai-Processing-Ms", "Openai-Version"} {
		if value := headers.Get(key); value != "" {
			result[key] = value
		}
	}
	return result
}

func retryAfter(value string) time.Duration {
	if value == "" {
		return 0
	}
	if duration, err := time.ParseDuration(value + "s"); err == nil {
		return duration
	}
	if when, err := http.ParseTime(value); err == nil {
		return max(time.Until(when), 0)
	}
	return 0
}

func bounded(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "…"
}

func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func rawObject(value json.RawMessage, label string) (map[string]any, error) {
	if len(bytes.TrimSpace(value)) == 0 {
		return nil, nil
	}
	var object map[string]any
	if err := json.Unmarshal(value, &object); err != nil || object == nil {
		return nil, fmt.Errorf("openai: %s must be a JSON object", label)
	}
	return object, nil
}
