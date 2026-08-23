package google

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

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

type Config struct {
	APIKey           string
	BaseURL          string
	Headers          http.Header
	Client           *http.Client
	Retry            llmux.RetryPolicy
	ProviderName     string
	AllowEmptyAPIKey bool
}

type Provider struct{ config Config }

type model struct {
	provider *Provider
	id       string
}

func New(config Config) (*Provider, error) {
	if config.APIKey == "" && !config.AllowEmptyAPIKey {
		return nil, errors.New("google: API key is empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = defaultBaseURL
	}
	parsed, err := url.Parse(config.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("google: invalid base URL %q", config.BaseURL)
	}
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.Client == nil {
		config.Client = httpx.NewClient()
	}
	if config.ProviderName == "" {
		config.ProviderName = "google.generative-ai"
	}
	config.Headers = config.Headers.Clone()
	return &Provider{config: config}, nil
}

func (provider *Provider) Name() string { return provider.config.ProviderName }

func (provider *Provider) Descriptor() llmux.ProviderDescriptor {
	return llmux.ProviderDescriptor{
		Name:           provider.Name(),
		WireProtocols:  []string{"google-generate-content"},
		Authentication: []string{"api-key"},
	}
}

func (provider *Provider) LanguageModel(modelID string) (llmux.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("google: model ID is empty")
	}
	return &model{provider: provider, id: modelID}, nil
}

func (model *model) ModelID() string { return model.id }

func (model *model) Generate(ctx context.Context, request llmux.Request) (llmux.Result, error) {
	body, warnings, err := model.build(request)
	if err != nil {
		return llmux.Result{}, err
	}
	response, err := httpx.Do(ctx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.endpoint(false), Headers: model.headers(request.Options.Headers), Body: body, Retry: model.provider.config.Retry})
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
	result.Warnings = warnings
	result.Response.Headers = selectedHeaders(response.Header)
	if request.Options.IncludeRawChunks {
		result.Raw = append(json.RawMessage(nil), payload...)
	}
	return llmux.ConformResult(result)
}

func (model *model) Stream(ctx context.Context, request llmux.Request) (llmux.Stream, error) {
	body, warnings, err := model.build(request)
	if err != nil {
		return nil, err
	}
	streamCtx, cancel := context.WithCancel(ctx)
	response, err := httpx.Do(streamCtx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.endpoint(true), Headers: model.headers(request.Options.Headers), Body: body, Retry: model.provider.config.Retry})
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
			return nil, fmt.Errorf("google: expected text/event-stream, got %q", contentType)
		}
	}
	return llmux.ConformStream(newGeminiStream(streamCtx, cancel, response.Body, model.provider.Name(), llmux.ResponseMetadata{ModelID: model.id, Headers: selectedHeaders(response.Header)}, warnings, request.Options.IncludeRawChunks)), nil
}

func (model *model) endpoint(streaming bool) string {
	path := model.id
	if !strings.Contains(path, "/") {
		path = "models/" + path
	}
	operation := ":generateContent"
	if streaming {
		operation = ":streamGenerateContent?alt=sse"
	}
	return model.provider.config.BaseURL + "/" + path + operation
}

func (model *model) headers(overrides map[string]string) http.Header {
	headers := model.provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Content-Type", "application/json")
	if model.provider.config.APIKey != "" {
		headers.Set("X-Goog-Api-Key", model.provider.config.APIKey)
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

func (model *model) build(request llmux.Request) ([]byte, []string, error) {
	contents := make([]any, 0, len(request.Messages))
	systemParts := make([]any, 0)
	if request.Instructions != "" {
		systemParts = append(systemParts, map[string]any{"text": request.Instructions})
	}
	for _, message := range request.Messages {
		if message.Role == llmux.RoleSystem {
			for _, part := range message.Content {
				if part.Kind != llmux.ContentText {
					return nil, nil, fmt.Errorf("google: unsupported system content %q", part.Kind)
				}
				systemParts = append(systemParts, map[string]any{"text": part.Text})
			}
			continue
		}
		content, err := googleContent(message)
		if err != nil {
			return nil, nil, err
		}
		contents = append(contents, content)
	}
	body := map[string]any{"contents": contents}
	if len(systemParts) > 0 && !strings.Contains(strings.ToLower(model.id), "gemma") {
		body["systemInstruction"] = map[string]any{"parts": systemParts}
	}
	generation := map[string]any{}
	if request.Options.MaxOutputTokens != nil {
		generation["maxOutputTokens"] = *request.Options.MaxOutputTokens
	}
	if request.Options.Temperature != nil {
		generation["temperature"] = *request.Options.Temperature
	}
	if request.Options.TopP != nil {
		generation["topP"] = *request.Options.TopP
	}
	if request.Options.TopK != nil {
		generation["topK"] = *request.Options.TopK
	}
	if len(request.Options.StopSequences) > 0 {
		generation["stopSequences"] = request.Options.StopSequences
	}
	if request.Options.Seed != nil {
		generation["seed"] = *request.Options.Seed
	}
	if request.Options.ResponseFormat != nil {
		switch request.Options.ResponseFormat.Type {
		case "json", "json_object", "json_schema":
			generation["responseMimeType"] = "application/json"
			if len(request.Options.ResponseFormat.Schema) > 0 {
				var schema any
				if err := json.Unmarshal(request.Options.ResponseFormat.Schema, &schema); err != nil {
					return nil, nil, fmt.Errorf("google: invalid response schema: %w", err)
				}
				generation["responseSchema"] = schema
			}
		case "text":
			generation["responseMimeType"] = "text/plain"
		}
	}
	if request.Options.Reasoning != nil {
		thinking := map[string]any{}
		if request.Options.Reasoning.Effort == "none" {
			thinking["thinkingBudget"] = 0
		} else if request.Options.Reasoning.Effort != "" {
			thinking["includeThoughts"] = true
			thinking["thinkingLevel"] = strings.ToUpper(request.Options.Reasoning.Effort)
		}
		if len(thinking) > 0 {
			generation["thinkingConfig"] = thinking
		}
	}
	if len(generation) > 0 {
		body["generationConfig"] = generation
	}
	if len(request.Options.Tools) > 0 {
		declarations := make([]any, 0, len(request.Options.Tools))
		for _, tool := range request.Options.Tools {
			var schema any = map[string]any{}
			if len(tool.InputSchema) > 0 {
				if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
					return nil, nil, fmt.Errorf("google: invalid tool schema for %q: %w", tool.Name, err)
				}
			}
			declarations = append(declarations, map[string]any{"name": tool.Name, "description": tool.Description, "parameters": schema})
		}
		body["tools"] = []any{map[string]any{"functionDeclarations": declarations}}
		if request.Options.ToolChoice != nil {
			config := map[string]any{}
			switch request.Options.ToolChoice.Mode {
			case llmux.ToolChoiceNone:
				config["mode"] = "NONE"
			case llmux.ToolChoiceRequired:
				config["mode"] = "ANY"
			case llmux.ToolChoiceNamed:
				config["mode"] = "ANY"
				config["allowedFunctionNames"] = []string{request.Options.ToolChoice.Name}
			default:
				config["mode"] = "AUTO"
			}
			body["toolConfig"] = map[string]any{"functionCallingConfig": config}
		}
	}
	providerOptions, err := rawObject(request.Options.ProviderOptions[model.provider.Name()], "provider options")
	if err != nil {
		return nil, nil, err
	}
	if providerOptions == nil {
		providerOptions, err = rawObject(request.Options.ProviderOptions["google"], "google provider options")
		if err != nil {
			return nil, nil, err
		}
	}
	overrides, err := rawObject(request.Options.BodyOverrides, "body overrides")
	if err != nil {
		return nil, nil, err
	}
	for _, extra := range []map[string]any{providerOptions, overrides} {
		for key, value := range extra {
			if key == "contents" || key == "systemInstruction" {
				return nil, nil, fmt.Errorf("google: %q cannot be overridden", key)
			}
			body[key] = value
		}
	}
	warnings := make([]string, 0, 2)
	if request.Options.PresencePenalty != nil {
		warnings = append(warnings, "presence penalty is not supported by Google and was omitted")
	}
	if request.Options.FrequencyPenalty != nil {
		warnings = append(warnings, "frequency penalty is not supported by Google and was omitted")
	}
	payload, err := json.Marshal(body)
	return payload, warnings, err
}

func googleContent(message llmux.Message) (map[string]any, error) {
	role := "user"
	if message.Role == llmux.RoleAssistant {
		role = "model"
	} else if message.Role != llmux.RoleUser && message.Role != llmux.RoleTool {
		return nil, fmt.Errorf("google: unsupported role %q", message.Role)
	}
	parts := make([]any, 0, len(message.Content))
	if message.Role == llmux.RoleAssistant && len(message.ProviderState) > 0 {
		var replay []any
		if err := json.Unmarshal(message.ProviderState, &replay); err != nil {
			return nil, fmt.Errorf("google: invalid assistant provider state: %w", err)
		}
		parts = append(parts, replay...)
	}
	for _, part := range message.Content {
		switch part.Kind {
		case llmux.ContentText, llmux.ContentCommentary, llmux.ContentFinalAnswer:
			parts = append(parts, map[string]any{"text": part.Text})
		case llmux.ContentReasoning:
			block := map[string]any{"text": part.Text, "thought": true}
			if len(part.ProviderData) > 0 {
				var extra map[string]any
				if json.Unmarshal(part.ProviderData, &extra) == nil {
					for key, value := range extra {
						block[key] = value
					}
				}
			}
			parts = append(parts, block)
		case llmux.ContentImage, llmux.ContentAudio, llmux.ContentFile:
			if part.URL != "" {
				parts = append(parts, map[string]any{"fileData": map[string]any{"mimeType": part.MediaType, "fileUri": part.URL}})
			} else if len(part.Data) > 0 {
				parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": part.MediaType, "data": base64.StdEncoding.EncodeToString(part.Data)}})
			} else {
				return nil, fmt.Errorf("google: content %q has neither URL nor data", part.Kind)
			}
		case llmux.ContentToolCall:
			if part.ToolCall == nil || !json.Valid(part.ToolCall.Arguments) {
				return nil, errors.New("google: invalid tool call")
			}
			var args any
			_ = json.Unmarshal(part.ToolCall.Arguments, &args)
			parts = append(parts, map[string]any{"functionCall": map[string]any{"id": part.ToolCall.ID, "name": part.ToolCall.Name, "args": args}})
		case llmux.ContentToolResult:
			if part.ToolResult == nil {
				return nil, errors.New("google: nil tool result")
			}
			content := part.ToolResult.Content
			if content == "" && len(part.ToolResult.Structured) > 0 {
				content = string(part.ToolResult.Structured)
			}
			name := first(part.ToolResult.Name, part.ToolResult.ToolCallID)
			parts = append(parts, map[string]any{"functionResponse": map[string]any{"id": part.ToolResult.ToolCallID, "name": name, "response": map[string]any{"name": name, "content": content}}})
		default:
			return nil, fmt.Errorf("google: unsupported content kind %q", part.Kind)
		}
	}
	if len(parts) == 0 {
		parts = append(parts, map[string]any{"text": ""})
	}
	return map[string]any{"role": role, "parts": parts}, nil
}

type wireResponse struct {
	ResponseID     string          `json:"responseId"`
	ModelVersion   string          `json:"modelVersion"`
	Candidates     []wireCandidate `json:"candidates"`
	UsageMetadata  googleUsage     `json:"usageMetadata"`
	PromptFeedback json.RawMessage `json:"promptFeedback"`
}

type wireCandidate struct {
	Content struct {
		Parts []json.RawMessage `json:"parts"`
	} `json:"content"`
	FinishReason       string          `json:"finishReason"`
	FinishMessage      string          `json:"finishMessage"`
	GroundingMetadata  json.RawMessage `json:"groundingMetadata"`
	URLContextMetadata json.RawMessage `json:"urlContextMetadata"`
}

type googleUsage struct {
	PromptTokenCount        int  `json:"promptTokenCount"`
	CandidatesTokenCount    int  `json:"candidatesTokenCount"`
	TotalTokenCount         int  `json:"totalTokenCount"`
	CachedContentTokenCount *int `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      int  `json:"thoughtsTokenCount"`
}

func parseResult(payload []byte) (llmux.Result, error) {
	var response wireResponse
	if err := json.Unmarshal(payload, &response); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid Google response: %w", err)
	}
	if len(response.Candidates) == 0 {
		return llmux.Result{}, errors.New("Google response has no candidates")
	}
	candidate := response.Candidates[0]
	result := llmux.Result{Response: llmux.ResponseMetadata{ID: response.ResponseID, ModelID: response.ModelVersion}, Usage: convertUsage(response.UsageMetadata), ProviderState: mustMarshal(candidate.Content.Parts)}
	if err := appendParts(&result, candidate.Content.Parts); err != nil {
		return llmux.Result{}, err
	}
	appendSources(&result, candidate.GroundingMetadata, candidate.URLContextMetadata)
	result.FinishReason, result.RawFinishReason = finishReason(candidate.FinishReason, len(result.ToolCalls) > 0)
	return result, nil
}

func appendParts(result *llmux.Result, parts []json.RawMessage) error {
	for _, raw := range parts {
		var part struct {
			Text             *string `json:"text"`
			Thought          bool    `json:"thought"`
			ThoughtSignature string  `json:"thoughtSignature"`
			FunctionCall     *struct {
				ID   string          `json:"id"`
				Name string          `json:"name"`
				Args json.RawMessage `json:"args"`
			} `json:"functionCall"`
			InlineData *struct {
				MimeType string `json:"mimeType"`
				Data     string `json:"data"`
			} `json:"inlineData"`
			FileData *struct {
				MimeType string `json:"mimeType"`
				FileURI  string `json:"fileUri"`
			} `json:"fileData"`
		}
		if err := json.Unmarshal(raw, &part); err != nil {
			return fmt.Errorf("invalid Google content part: %w", err)
		}
		if part.Text != nil {
			if part.Thought {
				result.Reasoning += *part.Text
				providerData, _ := json.Marshal(map[string]any{"thoughtSignature": part.ThoughtSignature})
				result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentReasoning, Text: *part.Text, ProviderData: providerData})
			} else {
				result.Text += *part.Text
				result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentText, Text: *part.Text})
			}
		}
		if part.FunctionCall != nil {
			arguments := part.FunctionCall.Args
			if len(arguments) == 0 {
				arguments = json.RawMessage(`{}`)
			}
			if !json.Valid(arguments) {
				return fmt.Errorf("Google tool call %q has invalid arguments", part.FunctionCall.ID)
			}
			call := llmux.ToolCall{ID: part.FunctionCall.ID, Name: part.FunctionCall.Name, Arguments: arguments}
			result.ToolCalls = append(result.ToolCalls, call)
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
		}
		if part.InlineData != nil {
			data, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
			if err != nil {
				return fmt.Errorf("invalid Google inline data: %w", err)
			}
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentFile, Data: data, MediaType: part.InlineData.MimeType})
		}
		if part.FileData != nil {
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentFile, URL: part.FileData.FileURI, MediaType: part.FileData.MimeType})
		}
	}
	return nil
}

func appendSources(result *llmux.Result, grounding, urlContext json.RawMessage) {
	seen := make(map[string]bool)
	var grounded struct {
		Chunks []struct {
			Web *struct {
				URI   string `json:"uri"`
				Title string `json:"title"`
			} `json:"web"`
		} `json:"groundingChunks"`
	}
	_ = json.Unmarshal(grounding, &grounded)
	for index, chunk := range grounded.Chunks {
		if chunk.Web != nil && chunk.Web.URI != "" && !seen[chunk.Web.URI] {
			seen[chunk.Web.URI] = true
			source := &llmux.Source{ID: fmt.Sprintf("grounding-%d", index), URL: chunk.Web.URI, Title: chunk.Web.Title}
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentSource, Source: source})
		}
	}
	var contextual struct {
		Metadata []struct {
			URL          string `json:"url"`
			RetrievedURL string `json:"retrievedUrl"`
		} `json:"urlMetadata"`
	}
	_ = json.Unmarshal(urlContext, &contextual)
	for index, item := range contextual.Metadata {
		uri := first(item.RetrievedURL, item.URL)
		if uri != "" && !seen[uri] {
			seen[uri] = true
			source := &llmux.Source{ID: fmt.Sprintf("url-context-%d", index), URL: uri}
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentSource, Source: source})
		}
	}
}

func finishReason(raw string, toolCalls bool) (llmux.FinishReason, string) {
	if toolCalls {
		return llmux.FinishToolCalls, raw
	}
	switch raw {
	case "STOP", "FINISH_REASON_UNSPECIFIED":
		return llmux.FinishStop, raw
	case "MAX_TOKENS":
		return llmux.FinishLength, raw
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return llmux.FinishContent, raw
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL":
		return llmux.FinishError, raw
	default:
		return llmux.FinishUnknown, raw
	}
}

func convertUsage(usage googleUsage) llmux.Usage {
	cached := 0
	if usage.CachedContentTokenCount != nil {
		cached = *usage.CachedContentTokenCount
	}
	return llmux.NormalizeUsage(llmux.Usage{
		InputTokens:               usage.PromptTokenCount,
		CachedInputTokens:         cached,
		CachedInputTokensReported: usage.CachedContentTokenCount != nil,
		OutputTokens:              llmux.SaturatingTokenSum(usage.CandidatesTokenCount, usage.ThoughtsTokenCount),
		ReasoningTokens:           usage.ThoughtsTokenCount,
		TotalTokens:               usage.TotalTokenCount,
	})
}

func (model *model) responseError(response *http.Response) error {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	var envelope struct {
		Error struct {
			Status  string `json:"status"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(payload, &envelope)
	error := &llmux.ProviderError{Provider: model.provider.Name(), Kind: llmux.ErrorKindForStatus(response.StatusCode), Code: envelope.Error.Status, StatusCode: response.StatusCode, Message: first(envelope.Error.Message, strings.TrimSpace(string(payload)))}
	if json.Valid(payload) {
		error.Raw = append(json.RawMessage(nil), payload...)
	}
	return error
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
	for _, name := range []string{"X-Request-Id", "X-Goog-Request-Info"} {
		if value := headers.Get(name); value != "" {
			result[name] = value
		}
	}
	return result
}

func rawObject(raw json.RawMessage, label string) (map[string]any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return nil, fmt.Errorf("google: %s must be a JSON object", label)
	}
	return object, nil
}

func mustMarshal(value any) json.RawMessage {
	payload, _ := json.Marshal(value)
	return payload
}

func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
