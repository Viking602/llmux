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
	ID               string              `json:"id"`
	DisplayName      string              `json:"displayName"`
	BaseURL          string              `json:"baseURL"`
	EnvKey           string              `json:"envKey"`
	Behavior         Behavior            `json:"behavior"`
	Protocol         Protocol            `json:"protocol,omitempty"`
	ExtraProtocols   []Protocol          `json:"extraProtocols,omitempty"`
	ProtocolBaseURL  map[Protocol]string `json:"protocolBaseURL,omitempty"`
	APIKeyHeader     string              `json:"apiKeyHeader,omitempty"`
	APIKeyPrefix     string              `json:"apiKeyPrefix,omitempty"`
	AllowEmptyAPIKey bool                `json:"allowEmptyAPIKey,omitempty"`
}

type Config struct {
	APIKey           string
	BaseURL          string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	AllowEmptyAPIKey bool
	// Protocol selects one advertised wire protocol for this provider.
	// Empty uses the profile default (Protocol, or chat-completions).
	Protocol Protocol
	// DefaultMaxOutputTokens is forwarded to Anthropic-protocol providers as
	// their provider-level max_tokens default when a request omits
	// CallOptions.MaxOutputTokens. Zero keeps the Anthropic package default.
	DefaultMaxOutputTokens int
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

func (profile Profile) Protocols() []Protocol {
	primary := profile.Protocol
	if primary == "" {
		primary = ProtocolChatCompletions
	}
	seen := map[Protocol]bool{primary: true}
	result := []Protocol{primary}
	for _, protocol := range profile.ExtraProtocols {
		if protocol == "" || seen[protocol] {
			continue
		}
		seen[protocol] = true
		result = append(result, protocol)
	}
	return result
}

func (profile Profile) resolveProtocol(requested Protocol) (Protocol, error) {
	supported := profile.Protocols()
	if requested == "" {
		return supported[0], nil
	}
	for _, protocol := range supported {
		if protocol == requested {
			return requested, nil
		}
	}
	return "", fmt.Errorf("provider compat: %s does not support protocol %q", profile.ID, requested)
}

func (profile Profile) baseURLFor(protocol Protocol) string {
	if profile.ProtocolBaseURL != nil {
		if override := strings.TrimSpace(profile.ProtocolBaseURL[protocol]); override != "" {
			return override
		}
	}
	return profile.BaseURL
}

func New(id string, config Config) (llmux.Provider, error) {
	profile, ok := Lookup(id)
	if !ok {
		return nil, errors.New("provider compat: unknown provider " + id)
	}
	protocol, err := profile.resolveProtocol(config.Protocol)
	if err != nil {
		return nil, err
	}
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(profile.EnvKey)
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = profile.baseURLFor(protocol)
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
	if protocol == ProtocolAnthropic {
		return anthropic.New(anthropic.Config{
			APIKey: apiKey, BaseURL: baseURL, Headers: config.Headers, Client: config.Client, Retry: config.Retry,
			ProviderName: profile.ID, AllowEmptyAPIKey: allowEmptyAPIKey,
			APIKeyHeader: profile.APIKeyHeader, APIKeyPrefix: profile.APIKeyPrefix,
			DefaultMaxOutputTokens: config.DefaultMaxOutputTokens,
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
	if protocol == ProtocolResponses {
		wireAPI = openai.Responses
	} else if protocol != ProtocolChatCompletions {
		return nil, fmt.Errorf("provider compat: %s uses unsupported protocol %q", profile.ID, protocol)
	}
	return openai.New(openai.Config{
		APIKey: apiKey, BaseURL: baseURL, Headers: config.Headers, Client: config.Client, Retry: config.Retry,
		ProviderName: profile.ID, AllowEmptyAPIKey: allowEmptyAPIKey, Profile: &behavior, WireAPI: wireAPI,
		APIKeyHeader: profile.APIKeyHeader, APIKeyPrefix: profile.APIKeyPrefix,
	})
}
