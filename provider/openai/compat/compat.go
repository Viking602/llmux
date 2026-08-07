// Package compat exposes llmux's protocol-compatible provider registry.
package compat

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/anthropic"
	"github.com/Viking602/llmux/provider/openai"
)

type Behavior string
type Protocol string

const (
	BehaviorFull     Behavior = "full"
	BehaviorGroq     Behavior = "groq"
	BehaviorDeepSeek Behavior = "deepseek"

	ProtocolChatCompletions Protocol = "chat-completions"
	ProtocolResponses       Protocol = "responses"
	ProtocolAnthropic       Protocol = "anthropic-messages"
)

type Profile struct {
	ID               string   `json:"id"`
	DisplayName      string   `json:"displayName"`
	BaseURL          string   `json:"baseURL"`
	EnvKey           string   `json:"envKey"`
	Behavior         Behavior `json:"behavior"`
	Protocol         Protocol `json:"protocol,omitempty"`
	APIKeyHeader     string   `json:"apiKeyHeader,omitempty"`
	APIKeyPrefix     string   `json:"apiKeyPrefix,omitempty"`
	AllowEmptyAPIKey bool     `json:"allowEmptyAPIKey,omitempty"`
}

type Config struct {
	APIKey           string
	BaseURL          string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	AllowEmptyAPIKey bool
}

func Lookup(id string) (Profile, bool) {
	id = normalizeProviderID(id)
	profile, ok := profiles[id]
	return profile, ok
}

// normalizeProviderID lowercases ids and maps underscores to hyphens so
// callers matching models.dev-style provider ids stay compatible with older
// underscore spellings.
func normalizeProviderID(id string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(id)), "_", "-")
}

func All() []Profile {
	result := make([]Profile, 0, len(profiles))
	for _, profile := range profiles {
		result = append(result, profile)
	}
	sort.Slice(result, func(left, right int) bool { return result[left].ID < result[right].ID })
	return result
}

func New(id string, config Config) (llmux.Provider, error) {
	profile, ok := Lookup(id)
	if !ok {
		return nil, errors.New("provider compat: unknown provider " + id)
	}
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(profile.EnvKey)
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = profile.BaseURL
	}
	if strings.TrimSpace(baseURL) == "" {
		return nil, fmt.Errorf("provider compat: %s requires an explicit base URL", profile.ID)
	}
	if strings.Contains(baseURL, "{CLOUDFLARE_ACCOUNT_ID}") {
		accountID := strings.TrimSpace(os.Getenv("CLOUDFLARE_ACCOUNT_ID"))
		if accountID == "" {
			return nil, fmt.Errorf("provider compat: %s requires CLOUDFLARE_ACCOUNT_ID or an explicit base URL", profile.ID)
		}
		baseURL = strings.ReplaceAll(baseURL, "{CLOUDFLARE_ACCOUNT_ID}", url.PathEscape(accountID))
	}
	allowEmptyAPIKey := config.AllowEmptyAPIKey || profile.AllowEmptyAPIKey
	if profile.Protocol == ProtocolAnthropic {
		return anthropic.New(anthropic.Config{
			APIKey: apiKey, BaseURL: baseURL, Headers: config.Headers, Client: config.Client, Retry: config.Retry,
			ProviderName: profile.ID, AllowEmptyAPIKey: allowEmptyAPIKey,
			APIKeyHeader: profile.APIKeyHeader, APIKeyPrefix: profile.APIKeyPrefix,
		})
	}
	behavior := openai.FullProfile()
	switch profile.Behavior {
	case BehaviorGroq:
		behavior.SupportsTopK = false
		behavior.StreamUsageKey = "x_groq"
	case BehaviorDeepSeek:
		behavior.DeepSeek = true
	}
	wireAPI := openai.ChatCompletions
	if profile.Protocol == ProtocolResponses {
		wireAPI = openai.Responses
	} else if profile.Protocol != "" && profile.Protocol != ProtocolChatCompletions {
		return nil, fmt.Errorf("provider compat: %s uses unsupported protocol %q", profile.ID, profile.Protocol)
	}
	return openai.New(openai.Config{
		APIKey: apiKey, BaseURL: baseURL, Headers: config.Headers, Client: config.Client, Retry: config.Retry,
		ProviderName: profile.ID, AllowEmptyAPIKey: allowEmptyAPIKey, Profile: &behavior, WireAPI: wireAPI,
		APIKeyHeader: profile.APIKeyHeader, APIKeyPrefix: profile.APIKeyPrefix,
	})
}
