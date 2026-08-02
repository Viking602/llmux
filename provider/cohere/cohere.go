package cohere

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

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

type Config struct {
	APIKey  string
	BaseURL string
	Headers http.Header
	Client  *http.Client
	Retry   llmux.RetryPolicy
}

type Provider struct{ config Config }

type model struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if config.APIKey == "" {
		return nil, errors.New("cohere: API key is empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://api.cohere.com/v2"
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("cohere: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	config.Headers = config.Headers.Clone()
	return &Provider{config: config}, nil
}

func (provider *Provider) Name() string { return "cohere" }

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("cohere: model ID is empty")
	}
	return &model{provider: provider, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	body, err := model.build(request, false)
	if err != nil {
		return llmux.Result{}, err
	}
	response, err := httpx.Do(ctx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.provider.config.BaseURL + "/chat", Headers: model.headers(request.Options.Headers), Body: body, Retry: model.provider.config.Retry})
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
		return llmux.Result{}, &llmux.ProviderError{Provider: "cohere", Kind: llmux.ErrorStream, Message: err.Error(), Cause: err}
	}
	if request.Options.IncludeRawChunks {
		result.Raw = append(json.RawMessage(nil), payload...)
	}
	return result, nil
}

func (model *model) Stream(ctx context.Context, request llmux.Request) (llmux.Stream, error) {
	body, err := model.build(request, true)
	if err != nil {
		return nil, err
	}
	streamCtx, cancel := context.WithCancel(ctx)
	response, err := httpx.Do(streamCtx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.provider.config.BaseURL + "/chat", Headers: model.headers(request.Options.Headers), Body: body, Retry: model.provider.config.Retry})
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
			return nil, fmt.Errorf("cohere: expected text/event-stream, got %q", contentType)
		}
	}
	return newChatStream(streamCtx, cancel, response.Body, request.Options.IncludeRawChunks), nil
}

func (model *model) headers(overrides map[string]string) http.Header {
	headers := model.provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Authorization", "Bearer "+model.provider.config.APIKey)
	headers.Set("Content-Type", "application/json")
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
	messages := make([]any, 0, len(request.Messages)+1)
	documents := make([]any, 0)
	if request.Instructions != "" {
		messages = append(messages, map[string]any{"role": "system", "content": request.Instructions})
	}
	for _, message := range request.Messages {
		wire, docs, err := cohereMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, wire...)
		documents = append(documents, docs...)
	}
	body := map[string]any{"model": model.id, "messages": messages}
	if streaming {
		body["stream"] = true
	}
	if len(documents) > 0 {
		body["documents"] = documents
	}
	options := request.Options
	if options.MaxOutputTokens != nil {
		body["max_tokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		body["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		body["p"] = *options.TopP
	}
	if options.TopK != nil {
		body["k"] = *options.TopK
	}
	if options.Seed != nil {
		body["seed"] = *options.Seed
	}
	if len(options.StopSequences) > 0 {
		body["stop_sequences"] = options.StopSequences
	}
	if options.FrequencyPenalty != nil {
		body["frequency_penalty"] = *options.FrequencyPenalty
	}
	if options.PresencePenalty != nil {
		body["presence_penalty"] = *options.PresencePenalty
	}
	if options.ResponseFormat != nil && options.ResponseFormat.Type != "text" {
		format := map[string]any{"type": "json_object"}
		if len(options.ResponseFormat.Schema) > 0 {
			var schema any
			if err := json.Unmarshal(options.ResponseFormat.Schema, &schema); err != nil {
				return nil, fmt.Errorf("cohere: invalid response schema: %w", err)
			}
			format["json_schema"] = schema
		}
		body["response_format"] = format
	}
	if len(options.Tools) > 0 {
		tools := make([]any, 0, len(options.Tools))
		for _, tool := range options.Tools {
			if options.ToolChoice != nil && options.ToolChoice.Mode == llmux.ToolChoiceNamed && tool.Name != options.ToolChoice.Name {
				continue
			}
			var schema any = map[string]any{}
			if len(tool.InputSchema) > 0 {
				if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
					return nil, fmt.Errorf("cohere: invalid tool schema: %w", err)
				}
			}
			tools = append(tools, map[string]any{"type": "function", "function": map[string]any{"name": tool.Name, "description": tool.Description, "parameters": schema}})
		}
		body["tools"] = tools
		if options.ToolChoice != nil {
			switch options.ToolChoice.Mode {
			case llmux.ToolChoiceNone:
				body["tool_choice"] = "NONE"
			case llmux.ToolChoiceRequired, llmux.ToolChoiceNamed:
				body["tool_choice"] = "REQUIRED"
			}
		}
	}
	if options.Reasoning != nil {
		thinking := map[string]any{"type": "disabled"}
		if options.Reasoning.Effort != "none" && options.Reasoning.Effort != "" {
			percentage := map[string]float64{"minimal": .02, "low": .1, "medium": .3, "high": .6, "xhigh": .9}[options.Reasoning.Effort]
			maxTokens := 32768
			if options.MaxOutputTokens != nil {
				maxTokens = *options.MaxOutputTokens
			}
			thinking = map[string]any{"type": "enabled", "token_budget": min(max(maxTokens*int(percentage*100)/100, 1024), maxTokens)}
		}
		body["thinking"] = thinking
	}
	for _, raw := range []json.RawMessage{options.ProviderOptions["cohere"], options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if err := json.Unmarshal(raw, &extra); err != nil || extra == nil {
			return nil, errors.New("cohere: provider options and body overrides must be JSON objects")
		}
		for key, value := range extra {
			if key == "model" || key == "messages" || key == "stream" {
				return nil, fmt.Errorf("cohere: %q cannot be overridden", key)
			}
			body[key] = value
		}
	}
	return json.Marshal(body)
}

func cohereMessage(message llmux.Message) ([]any, []any, error) {
	role := string(message.Role)
	if message.Role == llmux.RoleTool {
		messages := make([]any, 0)
		for _, part := range message.Content {
			if part.Kind == llmux.ContentToolResult && part.ToolResult != nil {
				content := part.ToolResult.Content
				if content == "" && len(part.ToolResult.Structured) > 0 {
					content = string(part.ToolResult.Structured)
				}
				messages = append(messages, map[string]any{"role": "tool", "content": content, "tool_call_id": part.ToolResult.ToolCallID})
			}
		}
		if len(messages) == 0 {
			return nil, nil, errors.New("cohere: tool message has no tool result")
		}
		return messages, nil, nil
	}
	content := make([]any, 0)
	documents := make([]any, 0)
	toolCalls := make([]any, 0)
	hasImage := false
	for _, part := range message.Content {
		switch part.Kind {
		case llmux.ContentText:
			content = append(content, map[string]any{"type": "text", "text": part.Text})
		case llmux.ContentImage:
			hasImage = true
			imageURL := part.URL
			if imageURL == "" && len(part.Data) > 0 {
				imageURL = "data:" + part.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
			}
			content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}})
		case llmux.ContentFile:
			if strings.HasPrefix(part.MediaType, "image/") {
				hasImage = true
				content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:" + part.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)}})
			} else {
				documents = append(documents, map[string]any{"data": map[string]any{"text": string(part.Data), "title": part.Filename}})
			}
		case llmux.ContentToolCall:
			if part.ToolCall == nil || !json.Valid(part.ToolCall.Arguments) {
				return nil, nil, errors.New("cohere: invalid tool call")
			}
			toolCalls = append(toolCalls, map[string]any{"id": part.ToolCall.ID, "type": "function", "function": map[string]any{"name": part.ToolCall.Name, "arguments": string(part.ToolCall.Arguments)}})
		}
	}
	wire := map[string]any{"role": role}
	if len(toolCalls) > 0 {
		wire["tool_calls"] = toolCalls
	} else if hasImage {
		wire["content"] = content
	} else {
		var text strings.Builder
		for _, item := range content {
			text.WriteString(item.(map[string]any)["text"].(string))
		}
		wire["content"] = text.String()
	}
	return []any{wire}, documents, nil
}

type wireResponse struct {
	GenerationID string `json:"generation_id"`
	Message      struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
		Citations []json.RawMessage `json:"citations"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
	Usage        struct {
		Tokens tokenPair `json:"tokens"`
	} `json:"usage"`
}

type tokenPair struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func parseResult(payload []byte) (llmux.Result, error) {
	var response wireResponse
	if err := json.Unmarshal(payload, &response); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid Cohere response: %w", err)
	}
	result := llmux.Result{Response: llmux.ResponseMetadata{ID: response.GenerationID}, Usage: convertUsage(response.Usage.Tokens), FinishReason: finishReason(response.FinishReason), RawFinishReason: response.FinishReason}
	for _, item := range response.Message.Content {
		if item.Type == "thinking" {
			result.Reasoning += item.Thinking
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentReasoning, Text: item.Thinking})
		} else {
			result.Text += item.Text
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentText, Text: item.Text})
		}
	}
	for _, tool := range response.Message.ToolCalls {
		arguments := json.RawMessage(strings.Replace(tool.Function.Arguments, "null", "{}", 1))
		if !json.Valid(arguments) {
			return llmux.Result{}, fmt.Errorf("Cohere tool call %q has invalid arguments", tool.ID)
		}
		call := llmux.ToolCall{ID: tool.ID, Name: tool.Function.Name, Arguments: arguments}
		result.ToolCalls = append(result.ToolCalls, call)
		result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
	}
	for index, raw := range response.Message.Citations {
		source := &llmux.Source{ID: fmt.Sprintf("citation-%d", index), Title: "Document"}
		var citation struct {
			Sources []struct {
				Document struct {
					Title string `json:"title"`
				} `json:"document"`
			} `json:"sources"`
		}
		if json.Unmarshal(raw, &citation) == nil && len(citation.Sources) > 0 && citation.Sources[0].Document.Title != "" {
			source.Title = citation.Sources[0].Document.Title
		}
		result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentSource, Source: source, ProviderData: raw})
	}
	return result, nil
}

func finishReason(raw string) llmux.FinishReason {
	switch raw {
	case "COMPLETE", "STOP_SEQUENCE":
		return llmux.FinishStop
	case "MAX_TOKENS":
		return llmux.FinishLength
	case "TOOL_CALL":
		return llmux.FinishToolCalls
	case "ERROR":
		return llmux.FinishError
	default:
		return llmux.FinishUnknown
	}
}

func convertUsage(tokens tokenPair) llmux.Usage {
	return llmux.Usage{InputTokens: tokens.InputTokens, OutputTokens: tokens.OutputTokens, TotalTokens: tokens.InputTokens + tokens.OutputTokens}
}

func (model *model) responseError(response *http.Response) error {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	var envelope struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(payload, &envelope)
	return &llmux.ProviderError{Provider: "cohere", Kind: llmux.ErrorKindForStatus(response.StatusCode), StatusCode: response.StatusCode, Message: first(envelope.Message, strings.TrimSpace(string(payload)))}
}

func (model *model) transportError(err error) error {
	kind := llmux.ErrorStream
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		kind = llmux.ErrorCancelled
	}
	return &llmux.ProviderError{Provider: "cohere", Kind: kind, Message: err.Error(), Cause: err}
}

func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
