package llmux

import (
	"context"
	"encoding/json"
	"time"
)

// Role identifies a message author on the provider-neutral wire model.
type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentKind identifies one message or result content block.
type ContentKind string

const (
	ContentText         ContentKind = "text"
	ContentReasoning    ContentKind = "reasoning"
	ContentImage        ContentKind = "image"
	ContentAudio        ContentKind = "audio"
	ContentFile         ContentKind = "file"
	ContentToolCall     ContentKind = "tool_call"
	ContentToolResult   ContentKind = "tool_result"
	ContentSource       ContentKind = "source"
	ContentProviderData ContentKind = "provider_data"
)

// ContentPart preserves typed content without forcing binary data through
// base64. ProviderData is an opaque provider-owned block that callers may
// replay but must not interpret.
type ContentPart struct {
	Kind         ContentKind     `json:"kind"`
	Text         string          `json:"text,omitempty"`
	Data         []byte          `json:"data,omitempty"`
	URL          string          `json:"url,omitempty"`
	MediaType    string          `json:"mediaType,omitempty"`
	Filename     string          `json:"filename,omitempty"`
	ToolCall     *ToolCall       `json:"toolCall,omitempty"`
	ToolResult   *ToolResult     `json:"toolResult,omitempty"`
	Source       *Source         `json:"source,omitempty"`
	ProviderData json.RawMessage `json:"providerData,omitempty"`
}

type Message struct {
	Role          Role            `json:"role"`
	Name          string          `json:"name,omitempty"`
	Content       []ContentPart   `json:"content,omitempty"`
	ProviderState json.RawMessage `json:"providerState,omitempty"`
}

func TextMessage(role Role, text string) Message {
	return Message{Role: role, Content: []ContentPart{{Kind: ContentText, Text: text}}}
}

type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Strict      bool            `json:"strict,omitempty"`
}

type ToolCall struct {
	ID               string          `json:"id"`
	Name             string          `json:"name"`
	Arguments        json.RawMessage `json:"arguments,omitempty"`
	ProviderExecuted *bool           `json:"providerExecuted,omitempty"`
	Dynamic          *bool           `json:"dynamic,omitempty"`
}

type ToolResult struct {
	ToolCallID  string          `json:"toolCallId"`
	Name        string          `json:"name,omitempty"`
	Content     string          `json:"content,omitempty"`
	Structured  json.RawMessage `json:"structured,omitempty"`
	IsError     bool            `json:"isError,omitempty"`
	Preliminary *bool           `json:"preliminary,omitempty"`
	Dynamic     *bool           `json:"dynamic,omitempty"`
}

type Source struct {
	ID        string `json:"id,omitempty"`
	URL       string `json:"url,omitempty"`
	Title     string `json:"title,omitempty"`
	MediaType string `json:"mediaType,omitempty"`
	Filename  string `json:"filename,omitempty"`
}

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "tool"
)

type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode"`
	Name string         `json:"name,omitempty"`
}

type ReasoningOptions struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type ResponseFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

// RetryPolicy applies only before any response content has been emitted.
// MaxAttempts includes the first request. Zero selects the default of three;
// one disables retries.
type RetryPolicy struct {
	MaxAttempts int           `json:"maxAttempts,omitempty"`
	BaseDelay   time.Duration `json:"baseDelay,omitempty"`
	MaxDelay    time.Duration `json:"maxDelay,omitempty"`
}

// CallOptions contains portable generation options. ProviderOptions are
// namespaced raw JSON values; BodyOverrides is the explicit escape hatch for
// provider fields not yet represented by the portable contract.
type CallOptions struct {
	MaxOutputTokens   *int                       `json:"maxOutputTokens,omitempty"`
	Temperature       *float64                   `json:"temperature,omitempty"`
	TopP              *float64                   `json:"topP,omitempty"`
	TopK              *int                       `json:"topK,omitempty"`
	StopSequences     []string                   `json:"stopSequences,omitempty"`
	PresencePenalty   *float64                   `json:"presencePenalty,omitempty"`
	FrequencyPenalty  *float64                   `json:"frequencyPenalty,omitempty"`
	Seed              *int64                     `json:"seed,omitempty"`
	Tools             []ToolDefinition           `json:"tools,omitempty"`
	ToolChoice        *ToolChoice                `json:"toolChoice,omitempty"`
	ParallelToolCalls *bool                      `json:"parallelToolCalls,omitempty"`
	Reasoning         *ReasoningOptions          `json:"reasoning,omitempty"`
	ResponseFormat    *ResponseFormat            `json:"responseFormat,omitempty"`
	Headers           map[string]string          `json:"headers,omitempty"`
	ProviderOptions   map[string]json.RawMessage `json:"providerOptions,omitempty"`
	BodyOverrides     json.RawMessage            `json:"bodyOverrides,omitempty"`
	IncludeRawChunks  bool                       `json:"includeRawChunks,omitempty"`
	MaxRetries        *int                       `json:"maxRetries,omitempty"`
}

type Request struct {
	Messages     []Message         `json:"messages"`
	Instructions string            `json:"instructions,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	Options      CallOptions       `json:"options,omitempty"`
}

type Usage struct {
	InputTokens                   int             `json:"inputTokens,omitempty"`
	CachedInputTokens             int             `json:"cachedInputTokens,omitempty"`
	CachedInputTokensReported     bool            `json:"cachedInputTokensReported,omitempty"`
	CacheWriteInputTokens         int             `json:"cacheWriteInputTokens,omitempty"`
	CacheWriteInputTokensReported bool            `json:"cacheWriteInputTokensReported,omitempty"`
	OutputTokens                  int             `json:"outputTokens,omitempty"`
	ReasoningTokens               int             `json:"reasoningTokens,omitempty"`
	TotalTokens                   int             `json:"totalTokens,omitempty"`
	Raw                           json.RawMessage `json:"raw,omitempty"`
}

type FinishReason string

const (
	FinishUnknown   FinishReason = "unknown"
	FinishStop      FinishReason = "stop"
	FinishLength    FinishReason = "length"
	FinishToolCalls FinishReason = "tool_calls"
	FinishContent   FinishReason = "content_filter"
	FinishError     FinishReason = "error"
	FinishCancelled FinishReason = "cancelled"
)

type ResponseMetadata struct {
	ID        string            `json:"id,omitempty"`
	ModelID   string            `json:"modelId,omitempty"`
	Timestamp time.Time         `json:"timestamp,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type Result struct {
	Content         []ContentPart    `json:"content,omitempty"`
	Text            string           `json:"text,omitempty"`
	Reasoning       string           `json:"reasoning,omitempty"`
	ToolCalls       []ToolCall       `json:"toolCalls,omitempty"`
	Usage           Usage            `json:"usage,omitempty"`
	FinishReason    FinishReason     `json:"finishReason"`
	RawFinishReason string           `json:"rawFinishReason,omitempty"`
	Warnings        []string         `json:"warnings,omitempty"`
	Response        ResponseMetadata `json:"response,omitempty"`
	ProviderState   json.RawMessage  `json:"providerState,omitempty"`
	Raw             json.RawMessage  `json:"raw,omitempty"`
}

// LanguageModel is the complete text-generation boundary. Implementations
// must be safe for concurrent use.
type LanguageModel interface {
	ModelID() string
	Generate(context.Context, Request) (Result, error)
	Stream(context.Context, Request) (Stream, error)
}

type Provider interface {
	Name() string
	LanguageModel(modelID string) (LanguageModel, error)
}

// ModelInfo is a portable entry from a provider List Models response.
// Raw retains the provider payload for callers that need extra fields.
type ModelInfo struct {
	ID          string          `json:"id"`
	DisplayName string          `json:"displayName,omitempty"`
	OwnedBy     string          `json:"ownedBy,omitempty"`
	Created     int64           `json:"created,omitempty"`
	Raw         json.RawMessage `json:"raw,omitempty"`
}

// ModelLister is an optional capability. Providers that expose a models
// discovery endpoint implement it; others leave the package helper returning
// ErrorUnsupported.
type ModelLister interface {
	ListModels(context.Context) ([]ModelInfo, error)
}

// ListModels discovers models when p implements ModelLister.
func ListModels(ctx context.Context, p Provider) ([]ModelInfo, error) {
	if p == nil {
		return nil, &ProviderError{Kind: ErrorInvalidRequest, Message: "provider is nil"}
	}
	lister, ok := p.(ModelLister)
	if !ok {
		return nil, &ProviderError{
			Provider: p.Name(),
			Kind:     ErrorUnsupported,
			Message:  "list models is not supported by this provider",
		}
	}
	return lister.ListModels(ctx)
}
