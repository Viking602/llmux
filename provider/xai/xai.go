package xai

import (
	"net/http"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/openai"
)

type Config struct {
	APIKey           string
	BaseURL          string
	Endpoint         string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	WireAPI          openai.WireAPI
	AllowEmptyAPIKey bool
}

func New(config Config) (*openai.Provider, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.x.ai/v1"
	}
	if config.WireAPI == "" {
		config.WireAPI = openai.Responses
	}
	profile := openai.FullProfile()
	profile.SupportsTopK = false
	profile.SupportsStop = false
	profile.SupportsPenalties = false
	profile.UsesMaxCompletionTokens = true
	profile.XAI = true
	return openai.New(openai.Config{APIKey: config.APIKey, BaseURL: config.BaseURL, Endpoint: config.Endpoint, Headers: config.Headers, Client: config.Client, Retry: config.Retry, ProviderName: "xai", WireAPI: config.WireAPI, AllowEmptyAPIKey: config.AllowEmptyAPIKey, Profile: &profile})
}
