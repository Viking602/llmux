package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

type CredentialsProvider func(context.Context) (Credentials, error)

type Config struct {
	Region              string
	Credentials         Credentials
	CredentialsProvider CredentialsProvider
	BearerToken         string
	BaseURL             string
	Headers             http.Header
	Client              *http.Client
	Retry               llmux.RetryPolicy
	Now                 func() time.Time
}

type Provider struct{ config Config }

type model struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if config.Region == "" {
		return nil, errors.New("bedrock: region is required")
	}
	if config.BearerToken == "" && config.CredentialsProvider == nil && (config.Credentials.AccessKeyID == "" || config.Credentials.SecretAccessKey == "") {
		return nil, errors.New("bedrock: bearer token or AWS credentials are required")
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://bedrock-runtime." + config.Region + ".amazonaws.com"
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("bedrock: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	if config.Now == nil {
		config.Now = time.Now
	}
	config.Headers = config.Headers.Clone()
	return &Provider{config: config}, nil
}

func (provider *Provider) Name() string { return "amazon-bedrock" }

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("bedrock: model ID is empty")
	}
	return &model{provider: provider, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	body, warnings, err := model.build(request)
	if err != nil {
		return llmux.Result{}, err
	}
	endpoint := model.endpoint(false)
	headers, err := model.headers(ctx, endpoint, body, request.Options.Headers)
	if err != nil {
		return llmux.Result{}, err
	}
	response, err := httpx.Do(ctx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: endpoint, Headers: headers, Body: body, Retry: model.provider.config.Retry})
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
		return llmux.Result{}, &llmux.ProviderError{Provider: "amazon-bedrock", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	result.Warnings = warnings
	if request.Options.IncludeRawChunks {
		result.Raw = append(json.RawMessage(nil), payload...)
	}
	return result, nil
}

func (model *model) Stream(ctx context.Context, request llmux.Request) (llmux.Stream, error) {
	body, warnings, err := model.build(request)
	if err != nil {
		return nil, err
	}
	endpoint := model.endpoint(true)
	headers, err := model.headers(ctx, endpoint, body, request.Options.Headers)
	if err != nil {
		return nil, err
	}
	streamCtx, cancel := context.WithCancel(ctx)
	response, err := httpx.Do(streamCtx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: endpoint, Headers: headers, Body: body, Retry: model.provider.config.Retry})
	if err != nil {
		cancel()
		return nil, model.transportError(err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		cancel()
		return nil, model.responseError(response)
	}
	return newConverseStream(streamCtx, cancel, response.Body, warnings, request.Options.IncludeRawChunks), nil
}

func (model *model) endpoint(streaming bool) string {
	operation := "converse"
	if streaming {
		operation = "converse-stream"
	}
	return model.provider.config.BaseURL + "/model/" + url.PathEscape(model.id) + "/" + operation
}

func (model *model) headers(ctx context.Context, endpoint string, body []byte, overrides map[string]string) (http.Header, error) {
	headers := model.provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Content-Type", "application/json")
	for name, value := range overrides {
		switch http.CanonicalHeaderKey(name) {
		case "Host", "Content-Length", "Transfer-Encoding", "Authorization", "X-Amz-Date", "X-Amz-Security-Token":
		default:
			headers.Set(name, value)
		}
	}
	if model.provider.config.BearerToken != "" {
		headers.Set("Authorization", "Bearer "+model.provider.config.BearerToken)
		return headers, nil
	}
	credentials := model.provider.config.Credentials
	if model.provider.config.CredentialsProvider != nil {
		var err error
		credentials, err = model.provider.config.CredentialsProvider(ctx)
		if err != nil {
			return nil, &llmux.ProviderError{Provider: "amazon-bedrock", Kind: llmux.ErrorAuthentication, Message: err.Error(), Cause: err}
		}
	}
	if credentials.AccessKeyID == "" || credentials.SecretAccessKey == "" {
		return nil, errors.New("bedrock: credentials provider returned empty credentials")
	}
	parsed, _ := url.Parse(endpoint)
	signV4(http.MethodPost, parsed, body, headers, credentials, model.provider.config.Region, model.provider.config.Now())
	return headers, nil
}

func (model *model) build(request llmux.Request) ([]byte, []string, error) {
	messages := make([]any, 0, len(request.Messages))
	system := make([]any, 0)
	if request.Instructions != "" {
		system = append(system, map[string]any{"text": request.Instructions})
	}
	for _, message := range request.Messages {
		if message.Role == llmux.RoleSystem {
			for _, part := range message.Content {
				if part.Kind != llmux.ContentText {
					return nil, nil, fmt.Errorf("bedrock: unsupported system content %q", part.Kind)
				}
				system = append(system, map[string]any{"text": part.Text})
			}
			continue
		}
		wire, err := bedrockMessage(message)
		if err != nil {
			return nil, nil, err
		}
		messages = append(messages, wire)
	}
	body := map[string]any{"messages": messages}
	if len(system) > 0 {
		body["system"] = system
	}
	inference := map[string]any{}
	if request.Options.MaxOutputTokens != nil {
		inference["maxTokens"] = *request.Options.MaxOutputTokens
	}
	if request.Options.Temperature != nil {
		inference["temperature"] = *request.Options.Temperature
	}
	if request.Options.TopP != nil {
		inference["topP"] = *request.Options.TopP
	}
	if len(request.Options.StopSequences) > 0 {
		inference["stopSequences"] = request.Options.StopSequences
	}
	if len(inference) > 0 {
		body["inferenceConfig"] = inference
	}
	if len(request.Options.Tools) > 0 {
		tools := make([]any, 0, len(request.Options.Tools))
		for _, tool := range request.Options.Tools {
			var schema any = map[string]any{}
			if len(tool.InputSchema) > 0 {
				if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
					return nil, nil, fmt.Errorf("bedrock: invalid tool schema: %w", err)
				}
			}
			tools = append(tools, map[string]any{"toolSpec": map[string]any{"name": tool.Name, "description": tool.Description, "inputSchema": map[string]any{"json": schema}}})
		}
		toolConfig := map[string]any{"tools": tools}
		if request.Options.ToolChoice != nil {
			switch request.Options.ToolChoice.Mode {
			case llmux.ToolChoiceAuto:
				toolConfig["toolChoice"] = map[string]any{"auto": map[string]any{}}
			case llmux.ToolChoiceRequired:
				toolConfig["toolChoice"] = map[string]any{"any": map[string]any{}}
			case llmux.ToolChoiceNamed:
				toolConfig["toolChoice"] = map[string]any{"tool": map[string]any{"name": request.Options.ToolChoice.Name}}
			}
		}
		body["toolConfig"] = toolConfig
	}
	additional := map[string]any{}
	if request.Options.TopK != nil {
		additional["top_k"] = *request.Options.TopK
	}
	if request.Options.Reasoning != nil && request.Options.Reasoning.Effort != "" {
		additional["reasoningConfig"] = map[string]any{"type": request.Options.Reasoning.Effort}
	}
	if len(additional) > 0 {
		body["additionalModelRequestFields"] = additional
	}
	for _, raw := range []json.RawMessage{request.Options.ProviderOptions["bedrock"], request.Options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if err := json.Unmarshal(raw, &extra); err != nil || extra == nil {
			return nil, nil, errors.New("bedrock: provider options and body overrides must be JSON objects")
		}
		for key, value := range extra {
			if key == "messages" || key == "system" {
				return nil, nil, fmt.Errorf("bedrock: %q cannot be overridden", key)
			}
			body[key] = value
		}
	}
	warnings := make([]string, 0, 4)
	if request.Options.Seed != nil {
		warnings = append(warnings, "seed is not supported by Bedrock Converse and was omitted")
	}
	if request.Options.PresencePenalty != nil {
		warnings = append(warnings, "presence penalty is not supported by Bedrock Converse and was omitted")
	}
	if request.Options.FrequencyPenalty != nil {
		warnings = append(warnings, "frequency penalty is not supported by Bedrock Converse and was omitted")
	}
	if request.Options.ResponseFormat != nil {
		warnings = append(warnings, "response format is model-specific on Bedrock and was omitted")
	}
	payload, err := json.Marshal(body)
	return payload, warnings, err
}

func bedrockMessage(message llmux.Message) (map[string]any, error) {
	role := string(message.Role)
	if message.Role == llmux.RoleTool {
		role = "user"
	}
	if role != "user" && role != "assistant" {
		return nil, fmt.Errorf("bedrock: unsupported role %q", message.Role)
	}
	content := make([]any, 0, len(message.Content))
	if message.Role == llmux.RoleAssistant && len(message.ProviderState) > 0 {
		var replay []any
		if err := json.Unmarshal(message.ProviderState, &replay); err != nil {
			return nil, fmt.Errorf("bedrock: invalid assistant provider state: %w", err)
		}
		content = append(content, replay...)
	}
	for _, part := range message.Content {
		switch part.Kind {
		case llmux.ContentText:
			content = append(content, map[string]any{"text": part.Text})
		case llmux.ContentReasoning:
			block := map[string]any{"reasoningContent": map[string]any{"reasoningText": map[string]any{"text": part.Text}}}
			content = append(content, block)
		case llmux.ContentImage:
			format := strings.TrimPrefix(part.MediaType, "image/")
			content = append(content, map[string]any{"image": map[string]any{"format": format, "source": map[string]any{"bytes": part.Data}}})
		case llmux.ContentFile:
			format := strings.TrimPrefix(part.MediaType, "application/")
			content = append(content, map[string]any{"document": map[string]any{"format": format, "name": first(part.Filename, "document"), "source": map[string]any{"bytes": part.Data}}})
		case llmux.ContentToolCall:
			if part.ToolCall == nil || !json.Valid(part.ToolCall.Arguments) {
				return nil, errors.New("bedrock: invalid tool call")
			}
			var input any
			_ = json.Unmarshal(part.ToolCall.Arguments, &input)
			content = append(content, map[string]any{"toolUse": map[string]any{"toolUseId": part.ToolCall.ID, "name": part.ToolCall.Name, "input": input}})
		case llmux.ContentToolResult:
			if part.ToolResult == nil {
				return nil, errors.New("bedrock: nil tool result")
			}
			resultContent := []any{map[string]any{"text": part.ToolResult.Content}}
			if len(part.ToolResult.Structured) > 0 {
				var value any
				_ = json.Unmarshal(part.ToolResult.Structured, &value)
				resultContent = []any{map[string]any{"json": value}}
			}
			status := "success"
			if part.ToolResult.IsError {
				status = "error"
			}
			content = append(content, map[string]any{"toolResult": map[string]any{"toolUseId": part.ToolResult.ToolCallID, "content": resultContent, "status": status}})
		default:
			return nil, fmt.Errorf("bedrock: unsupported content kind %q", part.Kind)
		}
	}
	return map[string]any{"role": role, "content": content}, nil
}

type wireUsage struct {
	InputTokens           int `json:"inputTokens"`
	OutputTokens          int `json:"outputTokens"`
	TotalTokens           int `json:"totalTokens"`
	CacheReadInputTokens  int `json:"cacheReadInputTokens"`
	CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
}

func parseResult(payload []byte) (llmux.Result, error) {
	var response struct {
		Output *struct {
			Message *struct {
				Content []json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"output"`
		StopReason string    `json:"stopReason"`
		Usage      wireUsage `json:"usage"`
	}
	if err := json.Unmarshal(payload, &response); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid Bedrock response: %w", err)
	}
	if response.Output == nil || response.Output.Message == nil {
		return llmux.Result{}, errors.New("Bedrock response has no output message")
	}
	result := llmux.Result{FinishReason: finishReason(response.StopReason), RawFinishReason: response.StopReason, Usage: usageResult(response.Usage), ProviderState: mustMarshal(response.Output.Message.Content)}
	if err := appendContent(&result, response.Output.Message.Content); err != nil {
		return llmux.Result{}, err
	}
	return result, nil
}

func appendContent(result *llmux.Result, blocks []json.RawMessage) error {
	for _, raw := range blocks {
		var block struct {
			Text    *string `json:"text"`
			ToolUse *struct {
				ToolUseID string          `json:"toolUseId"`
				Name      string          `json:"name"`
				Input     json.RawMessage `json:"input"`
			} `json:"toolUse"`
			Reasoning *struct {
				ReasoningText *struct {
					Text      string `json:"text"`
					Signature string `json:"signature"`
				} `json:"reasoningText"`
				RedactedContent json.RawMessage `json:"redactedContent"`
			} `json:"reasoningContent"`
		}
		if err := json.Unmarshal(raw, &block); err != nil {
			return err
		}
		if block.Text != nil {
			result.Text += *block.Text
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentText, Text: *block.Text})
		}
		if block.ToolUse != nil {
			if !json.Valid(block.ToolUse.Input) {
				return fmt.Errorf("Bedrock tool call %q has invalid input", block.ToolUse.ToolUseID)
			}
			call := llmux.ToolCall{ID: block.ToolUse.ToolUseID, Name: block.ToolUse.Name, Arguments: block.ToolUse.Input}
			result.ToolCalls = append(result.ToolCalls, call)
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
		}
		if block.Reasoning != nil && block.Reasoning.ReasoningText != nil {
			result.Reasoning += block.Reasoning.ReasoningText.Text
			providerData, _ := json.Marshal(map[string]any{"signature": block.Reasoning.ReasoningText.Signature})
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentReasoning, Text: block.Reasoning.ReasoningText.Text, ProviderData: providerData})
		}
	}
	return nil
}

func finishReason(raw string) llmux.FinishReason {
	switch raw {
	case "end_turn", "stop_sequence":
		return llmux.FinishStop
	case "max_tokens":
		return llmux.FinishLength
	case "tool_use":
		return llmux.FinishToolCalls
	case "content_filtered", "guardrail_intervened":
		return llmux.FinishContent
	default:
		return llmux.FinishUnknown
	}
}

func usageResult(usage wireUsage) llmux.Usage {
	return llmux.Usage{InputTokens: usage.InputTokens, CachedInputTokens: usage.CacheReadInputTokens, CacheWriteInputTokens: usage.CacheWriteInputTokens, OutputTokens: usage.OutputTokens, TotalTokens: usage.TotalTokens}
}

func (model *model) responseError(response *http.Response) error {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	var envelope struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	}
	_ = json.Unmarshal(payload, &envelope)
	return &llmux.ProviderError{Provider: "amazon-bedrock", Kind: llmux.ErrorKindForStatus(response.StatusCode), Code: envelope.Type, StatusCode: response.StatusCode, Message: first(envelope.Message, strings.TrimSpace(string(payload)))}
}

func (model *model) transportError(err error) error {
	kind := llmux.ErrorStream
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		kind = llmux.ErrorCancelled
	}
	return &llmux.ProviderError{Provider: "amazon-bedrock", Kind: kind, Message: err.Error(), Cause: err}
}

func mustMarshal(value any) json.RawMessage { payload, _ := json.Marshal(value); return payload }
func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
