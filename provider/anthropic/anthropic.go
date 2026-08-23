package anthropic

import (
	"context"
	"encoding/base64"
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

const (
	defaultBaseURL = "https://api.anthropic.com"
	defaultVersion = "2023-06-01"
	// DefaultMaxOutputTokens is used for Anthropic Messages max_tokens when the
	// request does not set CallOptions.MaxOutputTokens and Config also leaves
	// DefaultMaxOutputTokens unset. Anthropic requires max_tokens; hosts that
	// know a higher model limit should set one of those two fields.
	DefaultMaxOutputTokens = 4096
)

type Config struct {
	APIKey           string
	BaseURL          string
	Version          string
	Beta             []string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	ProviderName     string
	AllowEmptyAPIKey bool
	APIKeyHeader     string
	APIKeyPrefix     string
	// DefaultMaxOutputTokens is the provider-level max_tokens used when a
	// request omits CallOptions.MaxOutputTokens. Zero keeps
	// DefaultMaxOutputTokens (4096). Request-level MaxOutputTokens always wins.
	DefaultMaxOutputTokens int
}

type Provider struct{ config Config }

type model struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if strings.TrimSpace(config.APIKey) == "" && !config.AllowEmptyAPIKey {
		return nil, errors.New("anthropic: API key is empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = defaultBaseURL
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("anthropic: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimSuffix(strings.TrimRight(config.BaseURL, "/"), "/v1")
	if config.Version == "" {
		config.Version = defaultVersion
	}
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	if config.ProviderName == "" {
		config.ProviderName = "anthropic"
	}
	if config.APIKeyHeader == "" {
		config.APIKeyHeader = "X-Api-Key"
	}
	config.Headers = config.Headers.Clone()
	config.Beta = append([]string(nil), config.Beta...)
	return &Provider{config: config}, nil
}

func (provider *Provider) Name() string { return provider.config.ProviderName }

func (provider *Provider) Descriptor() llmux.ProviderDescriptor {
	return llmux.ProviderDescriptor{
		Name:           provider.Name(),
		WireProtocols:  []string{"anthropic-messages"},
		Authentication: []string{"x-api-key"},
	}
}

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("anthropic: model ID is empty")
	}
	return &model{provider: provider, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (provider *Provider) defaultMaxOutputTokens() int {
	if provider.config.DefaultMaxOutputTokens > 0 {
		return provider.config.DefaultMaxOutputTokens
	}
	return DefaultMaxOutputTokens
}

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	body, err := model.build(request, false)
	if err != nil {
		return llmux.Result{}, err
	}
	retry := model.provider.config.Retry
	if request.Options.MaxRetries != nil {
		retry.MaxAttempts = max(*request.Options.MaxRetries, 0) + 1
	}
	response, err := httpx.Do(ctx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.endpoint(), Headers: model.headers(request.Options.Headers), Body: body, Retry: retry})
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
	result, err := parseResult(payload)
	if err != nil {
		return llmux.Result{}, &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	result.Response.Headers = selectedHeaders(response.Header)
	if request.Options.IncludeRawChunks {
		result.Raw = append(json.RawMessage(nil), payload...)
	}
	return llmux.ConformResult(result)
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
	response, err := httpx.Do(streamCtx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.endpoint(), Headers: model.headers(request.Options.Headers), Body: body, Retry: retry})
	if err != nil {
		cancel()
		return nil, model.transportError(err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		cancel()
		return nil, model.responseError(response)
	}
	if contentType := response.Header.Get("Content-Type"); contentType != "" {
		mediaType, _, parseErr := mime.ParseMediaType(contentType)
		if parseErr != nil || mediaType != "text/event-stream" {
			_ = response.Body.Close()
			cancel()
			return nil, fmt.Errorf("anthropic: expected text/event-stream, got %q", contentType)
		}
	}
	return llmux.ConformStream(newMessageStream(streamCtx, cancel, response.Body, model.provider.Name(), llmux.ResponseMetadata{ModelID: model.id, Headers: selectedHeaders(response.Header)}, request.Options.IncludeRawChunks)), nil
}

func (model *model) endpoint() string { return model.provider.config.BaseURL + "/v1/messages" }

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
	headers.Set("Anthropic-Version", model.provider.config.Version)
	if len(model.provider.config.Beta) > 0 {
		headers.Set("Anthropic-Beta", strings.Join(model.provider.config.Beta, ","))
	}
	for name, value := range overrides {
		switch http.CanonicalHeaderKey(name) {
		case "Host", "Content-Length", "Transfer-Encoding":
		default:
			headers.Set(name, value)
		}
	}
	return headers
}

func (model *model) build(request llmux.Request, streaming bool) ([]byte, error) {
	system := make([]any, 0)
	if request.Instructions != "" {
		system = append(system, map[string]any{"type": "text", "text": request.Instructions})
	}
	messages := make([]any, 0, len(request.Messages))
	for _, message := range request.Messages {
		if message.Role == llmux.RoleSystem {
			for _, part := range message.Content {
				if part.Kind != llmux.ContentText {
					return nil, fmt.Errorf("anthropic: system content kind %q is unsupported", part.Kind)
				}
				system = append(system, map[string]any{"type": "text", "text": part.Text})
			}
			continue
		}
		wire, err := anthropicMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, wire)
	}
	maxTokens := model.provider.defaultMaxOutputTokens()
	if request.Options.MaxOutputTokens != nil {
		maxTokens = *request.Options.MaxOutputTokens
	}
	if maxTokens < 1 {
		return nil, errors.New("anthropic: max output tokens must be positive")
	}
	body := map[string]any{"model": model.id, "messages": messages, "max_tokens": maxTokens, "stream": streaming}
	if len(system) > 0 {
		body["system"] = system
	}
	applyOptions(body, request.Options)
	providerOptions, err := rawObject(request.Options.ProviderOptions[model.provider.Name()], "provider options")
	if err != nil {
		return nil, err
	}
	overrides, err := rawObject(request.Options.BodyOverrides, "body overrides")
	if err != nil {
		return nil, err
	}
	protected := map[string]bool{"model": true, "messages": true, "system": true, "stream": true}
	for _, extra := range []map[string]any{providerOptions, overrides} {
		for key, value := range extra {
			if protected[key] {
				return nil, fmt.Errorf("anthropic: %q cannot be overridden", key)
			}
			body[key] = value
		}
	}
	return json.Marshal(body)
}

func anthropicMessage(message llmux.Message) (map[string]any, error) {
	role := message.Role
	if role == llmux.RoleTool {
		role = llmux.RoleUser
	}
	if role != llmux.RoleUser && role != llmux.RoleAssistant {
		return nil, fmt.Errorf("anthropic: unsupported message role %q", message.Role)
	}
	blocks := make([]any, 0, len(message.Content))
	if message.Role == llmux.RoleAssistant && len(message.ProviderState) > 0 {
		var replay []any
		if err := json.Unmarshal(message.ProviderState, &replay); err != nil {
			return nil, fmt.Errorf("anthropic: invalid assistant provider state: %w", err)
		}
		blocks = append(blocks, replay...)
	}
	for _, part := range message.Content {
		switch part.Kind {
		case llmux.ContentText, llmux.ContentCommentary, llmux.ContentFinalAnswer:
			blocks = append(blocks, map[string]any{"type": "text", "text": part.Text})
		case llmux.ContentReasoning:
			block := map[string]any{"type": "thinking", "thinking": part.Text}
			if len(part.ProviderData) > 0 {
				var extra map[string]any
				if json.Unmarshal(part.ProviderData, &extra) == nil {
					for key, value := range extra {
						block[key] = value
					}
				}
			}
			blocks = append(blocks, block)
		case llmux.ContentImage, llmux.ContentFile:
			kind := "image"
			if part.Kind == llmux.ContentFile {
				kind = "document"
			}
			source := map[string]any{}
			if part.URL != "" {
				source["type"] = "url"
				source["url"] = part.URL
			} else if len(part.Data) > 0 {
				source["type"] = "base64"
				source["media_type"] = first(part.MediaType, "application/octet-stream")
				source["data"] = base64.StdEncoding.EncodeToString(part.Data)
			} else {
				return nil, fmt.Errorf("anthropic: %s has neither URL nor data", kind)
			}
			blocks = append(blocks, map[string]any{"type": kind, "source": source})
		case llmux.ContentToolCall:
			if part.ToolCall == nil || !json.Valid(part.ToolCall.Arguments) {
				return nil, errors.New("anthropic: invalid tool call")
			}
			var input any
			_ = json.Unmarshal(part.ToolCall.Arguments, &input)
			blocks = append(blocks, map[string]any{"type": "tool_use", "id": part.ToolCall.ID, "name": part.ToolCall.Name, "input": input})
		case llmux.ContentToolResult:
			if part.ToolResult == nil {
				return nil, errors.New("anthropic: nil tool result")
			}
			content := part.ToolResult.Content
			if content == "" && len(part.ToolResult.Structured) > 0 {
				content = string(part.ToolResult.Structured)
			}
			blocks = append(blocks, map[string]any{"type": "tool_result", "tool_use_id": part.ToolResult.ToolCallID, "content": content, "is_error": part.ToolResult.IsError})
		default:
			return nil, fmt.Errorf("anthropic: unsupported content kind %q", part.Kind)
		}
	}
	return map[string]any{"role": string(role), "content": blocks}, nil
}

func applyOptions(body map[string]any, options llmux.CallOptions) {
	if options.Temperature != nil {
		body["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		body["top_p"] = *options.TopP
	}
	if options.TopK != nil {
		body["top_k"] = *options.TopK
	}
	if len(options.StopSequences) > 0 {
		body["stop_sequences"] = options.StopSequences
	}
	if len(options.Tools) > 0 {
		tools := make([]any, 0, len(options.Tools))
		for _, tool := range options.Tools {
			var schema any = map[string]any{}
			if len(tool.InputSchema) > 0 {
				_ = json.Unmarshal(tool.InputSchema, &schema)
			}
			tools = append(tools, map[string]any{"name": tool.Name, "description": tool.Description, "input_schema": schema, "strict": tool.Strict})
		}
		body["tools"] = tools
	}
	if options.ToolChoice != nil && options.ToolChoice.Mode != llmux.ToolChoiceNone {
		choice := map[string]any{}
		switch options.ToolChoice.Mode {
		case llmux.ToolChoiceAuto:
			choice["type"] = "auto"
		case llmux.ToolChoiceRequired:
			choice["type"] = "any"
		case llmux.ToolChoiceNamed:
			choice["type"] = "tool"
			choice["name"] = options.ToolChoice.Name
		}
		if options.ParallelToolCalls != nil {
			choice["disable_parallel_tool_use"] = !*options.ParallelToolCalls
		}
		body["tool_choice"] = choice
	}
	if options.Reasoning != nil {
		if options.Reasoning.Effort == "none" {
			body["thinking"] = map[string]any{"type": "disabled"}
		} else if options.Reasoning.Effort != "" {
			body["thinking"] = map[string]any{"type": "adaptive"}
			body["output_config"] = map[string]any{"effort": options.Reasoning.Effort}
			delete(body, "temperature")
			delete(body, "top_p")
			delete(body, "top_k")
		}
	}
}

type response struct {
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	Content    json.RawMessage `json:"content"`
	StopReason string          `json:"stop_reason"`
	Usage      usage           `json:"usage"`
}

type usage struct {
	InputTokens              int  `json:"input_tokens"`
	OutputTokens             int  `json:"output_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	OutputTokensDetails      struct {
		ThinkingTokens int `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
}

func parseResult(payload []byte) (llmux.Result, error) {
	var wire response
	if err := json.Unmarshal(payload, &wire); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid Anthropic response: %w", err)
	}
	var blocks []struct {
		Type      string          `json:"type"`
		Text      string          `json:"text"`
		Thinking  string          `json:"thinking"`
		Signature string          `json:"signature"`
		Data      string          `json:"data"`
		ID        string          `json:"id"`
		Name      string          `json:"name"`
		Input     json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(wire.Content, &blocks); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid Anthropic content: %w", err)
	}
	result := llmux.Result{FinishReason: finishReason(wire.StopReason), RawFinishReason: wire.StopReason, Response: llmux.ResponseMetadata{ID: wire.ID, ModelID: wire.Model}, ProviderState: append(json.RawMessage(nil), wire.Content...), Usage: usageResult(wire.Usage)}
	for _, block := range blocks {
		switch block.Type {
		case "text":
			result.Text += block.Text
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentText, Text: block.Text})
		case "thinking":
			result.Reasoning += block.Thinking
			providerData, _ := json.Marshal(map[string]any{"signature": block.Signature})
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentReasoning, Text: block.Thinking, ProviderData: providerData})
		case "redacted_thinking":
			providerData, _ := json.Marshal(map[string]any{"type": block.Type, "data": block.Data})
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentProviderData, ProviderData: providerData})
		case "tool_use":
			if !json.Valid(block.Input) {
				return llmux.Result{}, fmt.Errorf("Anthropic tool call %q has invalid input", block.ID)
			}
			call := llmux.ToolCall{ID: block.ID, Name: block.Name, Arguments: block.Input}
			result.ToolCalls = append(result.ToolCalls, call)
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
		}
	}
	return result, nil
}

func finishReason(reason string) llmux.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return llmux.FinishStop
	case "max_tokens":
		return llmux.FinishLength
	case "tool_use":
		return llmux.FinishToolCalls
	case "refusal":
		return llmux.FinishContent
	default:
		return llmux.FinishUnknown
	}
}

func usageResult(value usage) llmux.Usage {
	cacheRead := intValue(value.CacheReadInputTokens)
	cacheWrite := intValue(value.CacheCreationInputTokens)
	input := llmux.SaturatingTokenSum(value.InputTokens, cacheRead, cacheWrite)
	return llmux.NormalizeUsage(llmux.Usage{
		InputTokens:                   input,
		CachedInputTokens:             cacheRead,
		CachedInputTokensReported:     value.CacheReadInputTokens != nil,
		CacheWriteInputTokens:         cacheWrite,
		CacheWriteInputTokensReported: value.CacheCreationInputTokens != nil,
		OutputTokens:                  value.OutputTokens,
		ReasoningTokens:               value.OutputTokensDetails.ThinkingTokens,
		TotalTokens:                   llmux.SaturatingTokenSum(input, value.OutputTokens),
	})
}

func intValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

func (model *model) responseError(response *http.Response) error {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := first(envelope.Error.Message, strings.TrimSpace(string(payload)))
	providerError := &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorKindForStatus(response.StatusCode), Code: envelope.Error.Type, StatusCode: response.StatusCode, Message: bounded(message, 8<<10), RetryAfter: retryAfter(response.Header.Get("Retry-After"))}
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
	for _, key := range []string{"Request-Id", "X-Request-Id", "Anthropic-Organization-Id"} {
		if value := headers.Get(key); value != "" {
			result[key] = value
		}
	}
	return result
}

func retryAfter(value string) time.Duration {
	if duration, err := time.ParseDuration(value + "s"); value != "" && err == nil {
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

func rawObject(raw json.RawMessage, label string) (map[string]any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return nil, fmt.Errorf("anthropic: %s must be a JSON object", label)
	}
	return object, nil
}
