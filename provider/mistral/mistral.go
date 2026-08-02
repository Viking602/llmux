package mistral

import (
	"net/http"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/openai"
)

type Config struct {
	APIKey  string
	BaseURL string
	Headers http.Header
	Client  *http.Client
	Retry   llmux.RetryPolicy
}

func New(config Config) (*openai.Provider, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.mistral.ai/v1"
	}
	profile := openai.FullProfile()
	profile.RequiredToolChoice = "any"
	return openai.New(openai.Config{APIKey: config.APIKey, BaseURL: config.BaseURL, Headers: config.Headers, Client: config.Client, Retry: config.Retry, ProviderName: "mistral", Profile: &profile, WireAPI: openai.ChatCompletions})
}
