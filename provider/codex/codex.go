// Package codex implements OpenAI Codex access through the Responses API.
package codex

import (
	"bytes"
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
	"github.com/Viking602/llmux/provider/openai"
)

const (
	APIBaseURL          = "https://api.openai.com/v1"
	SubscriptionBaseURL = "https://chatgpt.com/backend-api/codex"
	APIKeyEnvVar        = "CODEX_API_KEY"
	OAuthTokenURL       = "https://auth.openai.com/oauth/token"
)

// Mode selects the official API-key channel or the ChatGPT subscription channel.
type Mode string

const (
	ModeAPIKey       Mode = "api_key"
	ModeSubscription Mode = "subscription"
)

type Config struct {
	APIKey           string
	Mode             Mode
	BaseURL          string
	Endpoint         string
	ChatGPTAccountID string
	Originator       string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
}

type Provider struct {
	inner *openai.Provider
	mode  Mode
}

type model struct {
	inner llmux.LanguageModel
	mode  Mode
}

func New(config Config) (*Provider, error) {
	if config.Mode == "" {
		config.Mode = ModeAPIKey
	}
	if config.Mode != ModeAPIKey && config.Mode != ModeSubscription {
		return nil, fmt.Errorf("codex: unsupported mode %q", config.Mode)
	}
	if config.BaseURL == "" {
		if config.Mode == ModeSubscription {
			config.BaseURL = SubscriptionBaseURL
		} else {
			config.BaseURL = APIBaseURL
		}
	}

	headers := config.Headers.Clone()
	if config.Mode == ModeSubscription {
		if headers == nil {
			headers = make(http.Header)
		}
		originator := config.Originator
		if originator == "" {
			originator = "llmux"
		}
		if headers.Get("Originator") == "" {
			headers.Set("Originator", originator)
		}
		if config.ChatGPTAccountID != "" && headers.Get("ChatGPT-Account-Id") == "" {
			headers.Set("ChatGPT-Account-Id", config.ChatGPTAccountID)
		}
	}

	profile := openai.FullProfile()
	inner, err := openai.New(openai.Config{
		APIKey: config.APIKey, BaseURL: config.BaseURL, Endpoint: config.Endpoint,
		Headers: headers, Client: config.Client, Retry: config.Retry,
		WireAPI: openai.Responses, ProviderName: "codex", Profile: &profile,
	})
	if err != nil {
		return nil, err
	}
	return &Provider{inner: inner, mode: config.Mode}, nil
}

func (provider *Provider) Name() string { return "codex" }

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	inner, err := provider.inner.LanguageModel(modelID)
	if err != nil {
		return nil, err
	}
	return &model{inner: inner, mode: provider.mode}, nil
}

func (model *model) ModelID() string { return model.inner.ModelID() }

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	if model.mode == ModeAPIKey {
		return model.inner.Generate(ctx, request)
	}
	request, err := subscriptionRequest(request)
	if err != nil {
		return llmux.Result{}, err
	}
	stream, err := model.inner.Stream(ctx, request)
	if err != nil {
		return llmux.Result{}, mapSubscriptionError(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil {
		return llmux.Result{}, mapSubscriptionError(err)
	}
	return result, nil
}

func (model *model) Stream(ctx context.Context, request llmux.Request) (llmux.Stream, error) {
	if model.mode == ModeSubscription {
		var err error
		request, err = subscriptionRequest(request)
		if err != nil {
			return nil, err
		}
	}
	stream, err := model.inner.Stream(ctx, request)
	if err != nil && model.mode == ModeSubscription {
		return nil, mapSubscriptionError(err)
	}
	return stream, err
}

func subscriptionRequest(request llmux.Request) (llmux.Request, error) {
	overrides := make(map[string]json.RawMessage)
	if len(request.Options.BodyOverrides) > 0 {
		if err := json.Unmarshal(request.Options.BodyOverrides, &overrides); err != nil {
			return llmux.Request{}, fmt.Errorf("codex: invalid body overrides: %w", err)
		}
		if overrides == nil {
			overrides = make(map[string]json.RawMessage)
		}
	}
	overrides["store"] = json.RawMessage("false")
	raw, err := json.Marshal(overrides)
	if err != nil {
		return llmux.Request{}, fmt.Errorf("codex: encode body overrides: %w", err)
	}
	request.Options.BodyOverrides = raw
	return request, nil
}

func mapSubscriptionError(err error) error {
	var providerError *llmux.ProviderError
	if !errors.As(err, &providerError) || providerError.StatusCode != http.StatusUnauthorized {
		return err
	}
	mapped := *providerError
	mapped.Kind = llmux.ErrorTokenExpired
	if mapped.Code == "" {
		mapped.Code = "token_expired"
	}
	return &mapped
}

// Tokens contains credentials returned by the Codex OAuth refresh endpoint.
type Tokens struct {
	AccessToken      string  `json:"accessToken"`
	RefreshToken     string  `json:"refreshToken,omitempty"`
	ExpiresInSeconds *uint64 `json:"expiresInSeconds,omitempty"`
}

// RefreshConfig describes one stateless OAuth token refresh. Refresh never retries because
// refresh tokens rotate on use and replaying a request can invalidate the new token.
type RefreshConfig struct {
	RefreshToken string
	ClientID     string
	TokenURL     string
	Client       *http.Client
}

func Refresh(ctx context.Context, config RefreshConfig) (Tokens, error) {
	if strings.TrimSpace(config.RefreshToken) == "" {
		return Tokens{}, errors.New("codex: refresh token is empty")
	}
	if strings.TrimSpace(config.ClientID) == "" {
		return Tokens{}, errors.New("codex: client ID is empty")
	}
	if config.TokenURL == "" {
		config.TokenURL = OAuthTokenURL
	}
	parsed, err := url.Parse(config.TokenURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return Tokens{}, fmt.Errorf("codex: invalid token URL %q", config.TokenURL)
	}
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}

	payload, err := json.Marshal(map[string]string{
		"grant_type":    "refresh_token",
		"refresh_token": config.RefreshToken,
		"client_id":     config.ClientID,
	})
	if err != nil {
		return Tokens{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, config.TokenURL, bytes.NewReader(payload))
	if err != nil {
		return Tokens{}, err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")

	client := *config.Client
	client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	response, err := client.Do(request)
	if err != nil {
		kind := llmux.ErrorStream
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			kind = llmux.ErrorCancelled
		}
		return Tokens{}, &llmux.ProviderError{Provider: "codex", Kind: kind, Message: err.Error(), Cause: err}
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		return Tokens{}, &llmux.ProviderError{Provider: "codex", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}

	var envelope struct {
		AccessToken      string  `json:"access_token"`
		RefreshToken     string  `json:"refresh_token"`
		ExpiresInSeconds *uint64 `json:"expires_in"`
		Error            string  `json:"error"`
		ErrorDescription string  `json:"error_description"`
	}
	decodeErr := json.Unmarshal(body, &envelope)
	if response.StatusCode/100 != 2 {
		message := strings.TrimSpace(envelope.ErrorDescription)
		if message == "" {
			message = strings.TrimSpace(envelope.Error)
		}
		if message == "" {
			message = fmt.Sprintf("OAuth token endpoint returned HTTP %d", response.StatusCode)
		}
		return Tokens{}, &llmux.ProviderError{
			Provider: "codex", Kind: llmux.ErrorKindForStatus(response.StatusCode), Code: envelope.Error,
			StatusCode: response.StatusCode, Message: message,
		}
	}
	if decodeErr != nil {
		return Tokens{}, &llmux.ProviderError{Provider: "codex", Kind: llmux.ErrorUnknown, Code: "invalid_token_response", Message: decodeErr.Error(), Cause: decodeErr}
	}
	if envelope.AccessToken == "" {
		return Tokens{}, &llmux.ProviderError{Provider: "codex", Kind: llmux.ErrorUnknown, Code: "invalid_token_response", Message: "OAuth token response is missing access_token"}
	}
	return Tokens{AccessToken: envelope.AccessToken, RefreshToken: envelope.RefreshToken, ExpiresInSeconds: envelope.ExpiresInSeconds}, nil
}
