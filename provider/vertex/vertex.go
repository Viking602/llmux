package vertex

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/google"
)

type TokenSource func(context.Context) (string, error)

type Config struct {
	Project     string
	Location    string
	AccessToken string
	TokenSource TokenSource
	BaseURL     string
	Headers     http.Header
	Client      *http.Client
	Retry       llmux.RetryPolicy
}

type Provider struct {
	config   Config
	delegate *google.Provider
}

type model struct {
	provider *Provider
	delegate llmux.LanguageModel
	id       string
}

func New(config Config) (*Provider, error) {
	if config.Project == "" || config.Location == "" {
		return nil, errors.New("vertex: project and location are required")
	}
	if config.AccessToken == "" && config.TokenSource == nil {
		return nil, errors.New("vertex: access token or token source is required")
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://" + config.Location + "-aiplatform.googleapis.com/v1"
	}
	delegate, err := google.New(google.Config{BaseURL: config.BaseURL, Headers: config.Headers, Client: config.Client, Retry: config.Retry, ProviderName: "vertex", AllowEmptyAPIKey: true})
	if err != nil {
		return nil, err
	}
	return &Provider{config: config, delegate: delegate}, nil
}

func (provider *Provider) Name() string { return "vertex" }

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("vertex: model ID is empty")
	}
	path := "projects/" + provider.config.Project + "/locations/" + provider.config.Location + "/publishers/google/models/" + modelID
	delegate, err := provider.delegate.LanguageModel(path)
	if err != nil {
		return nil, err
	}
	return &model{provider: provider, delegate: delegate, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	request, err := model.authorize(ctx, request)
	if err != nil {
		return llmux.Result{}, err
	}
	return model.delegate.Generate(ctx, request)
}

func (model *model) Stream(ctx context.Context, request llmux.Request) (llmux.Stream, error) {
	request, err := model.authorize(ctx, request)
	if err != nil {
		return nil, err
	}
	return model.delegate.Stream(ctx, request)
}

func (model *model) authorize(ctx context.Context, request llmux.Request) (llmux.Request, error) {
	token := model.provider.config.AccessToken
	var err error
	if model.provider.config.TokenSource != nil {
		token, err = model.provider.config.TokenSource(ctx)
		if err != nil {
			return llmux.Request{}, &llmux.ProviderError{Provider: "vertex", Kind: llmux.ErrorAuthentication, Message: err.Error(), Cause: err}
		}
	}
	if token == "" {
		return llmux.Request{}, errors.New("vertex: token source returned an empty token")
	}
	headers := make(map[string]string, len(request.Options.Headers)+1)
	for key, value := range request.Options.Headers {
		headers[key] = value
	}
	headers["Authorization"] = "Bearer " + token
	request.Options.Headers = headers
	return request, nil
}
