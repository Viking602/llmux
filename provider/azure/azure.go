package azure

import (
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/openai"
)

type Config struct {
	APIKey      string
	ResourceURL string
	Deployment  string
	APIVersion  string
	Headers     http.Header
	Client      *http.Client
	Retry       llmux.RetryPolicy
	WireAPI     openai.WireAPI
}

func New(config Config) (*openai.Provider, error) {
	if config.APIKey == "" || config.ResourceURL == "" || config.Deployment == "" {
		return nil, errors.New("azure: API key, resource URL, and deployment are required")
	}
	if config.APIVersion == "" {
		config.APIVersion = "2025-04-01-preview"
	}
	if config.WireAPI == "" {
		config.WireAPI = openai.Responses
	}
	path := "/openai/deployments/" + url.PathEscape(config.Deployment) + "/chat/completions"
	if config.WireAPI == openai.Responses {
		path = "/openai/responses"
	}
	resource := strings.TrimRight(config.ResourceURL, "/")
	endpoint := resource + path + "?api-version=" + url.QueryEscape(config.APIVersion)
	// Data-plane catalog of accessible base models (not deployment names).
	listModelsURL := resource + "/openai/models?api-version=" + url.QueryEscape(config.APIVersion)
	return openai.New(openai.Config{
		APIKey: config.APIKey, BaseURL: config.ResourceURL, Endpoint: endpoint, APIKeyHeader: "api-key",
		Headers: config.Headers, Client: config.Client, Retry: config.Retry, WireAPI: config.WireAPI, ProviderName: "azure",
		ListModelsURL: listModelsURL,
	})
}
