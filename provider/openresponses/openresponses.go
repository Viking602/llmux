// Package openresponses implements a generic Open Responses endpoint.
package openresponses

import (
	"net/http"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/openai"
)

type Config struct {
	ProviderName     string
	APIKey           string
	Endpoint         string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	AllowEmptyAPIKey bool
}

func New(config Config) (*openai.Provider, error) {
	if config.ProviderName == "" {
		config.ProviderName = "open-responses"
	}
	return openai.New(openai.Config{
		APIKey: config.APIKey, BaseURL: endpointBase(config.Endpoint), Endpoint: config.Endpoint,
		Headers: config.Headers, Client: config.Client, Retry: config.Retry, WireAPI: openai.Responses,
		ProviderName: config.ProviderName, AllowEmptyAPIKey: config.AllowEmptyAPIKey,
	})
}

func endpointBase(endpoint string) string {
	if endpoint == "" {
		return "http://127.0.0.1:1234/v1"
	}
	return endpoint
}
