package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Viking602/llmux"
)

func (model *model) build(request llmux.Request, streaming bool) ([]byte, error) {
	var body map[string]any
	var err error
	if model.provider.config.WireAPI == Responses {
		body, err = model.responsesBody(request, streaming)
	} else {
		body, err = model.chatBody(request, streaming)
	}
	if err != nil {
		return nil, err
	}
	providerOptions, err := rawObject(request.Options.ProviderOptions[model.provider.Name()], "provider options")
	if err != nil {
		return nil, err
	}
	if providerOptions == nil && model.provider.Name() != "openai" {
		providerOptions, err = rawObject(request.Options.ProviderOptions["openai"], "openai provider options")
		if err != nil {
			return nil, err
		}
	}
	if model.provider.config.Profile.XAI {
		providerOptions = normalizeXAIOptions(providerOptions)
	}
	overrides, err := rawObject(request.Options.BodyOverrides, "body overrides")
	if err != nil {
		return nil, err
	}
	protected := map[string]bool{"model": true, "messages": true, "input": true, "stream": true}
	for _, extra := range []map[string]any{providerOptions, overrides} {
		for key, value := range extra {
			if protected[key] {
				return nil, fmt.Errorf("openai: %q cannot be overridden", key)
			}
			body[key] = value
		}
	}
	return json.Marshal(body)
}

func (model *model) chatBody(request llmux.Request, streaming bool) (map[string]any, error) {
	messages := make([]any, 0, len(request.Messages)+1)
	if request.Instructions != "" {
		messages = append(messages, map[string]any{"role": "system", "content": request.Instructions})
	}
	for _, message := range request.Messages {
		wire, err := chatMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, wire)
	}
	body := map[string]any{"model": model.id, "messages": messages, "stream": streaming}
	if streaming && model.provider.config.Profile.StreamUsageKey != "x_groq" {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	applyPortableOptions(body, request.Options, false, *model.provider.config.Profile, model.provider.Name(), model.id)
	if len(request.Metadata) > 0 {
		body["metadata"] = request.Metadata
	}
	return body, nil
}

func chatMessage(message llmux.Message) (map[string]any, error) {
	if message.Role == "" {
		return nil, fmt.Errorf("openai: message role is empty")
	}
	if message.Role == llmux.RoleTool {
		for _, part := range message.Content {
			if part.Kind == llmux.ContentToolResult && part.ToolResult != nil {
				content := part.ToolResult.Content
				if content == "" && len(part.ToolResult.Structured) > 0 {
					content = string(part.ToolResult.Structured)
				}
				return map[string]any{"role": "tool", "tool_call_id": part.ToolResult.ToolCallID, "content": content}, nil
			}
		}
		return nil, fmt.Errorf("openai: tool message has no tool result")
	}
	wire := map[string]any{"role": string(message.Role)}
	if message.Name != "" {
		wire["name"] = message.Name
	}
	content := make([]any, 0, len(message.Content))
	toolCalls := make([]any, 0)
	for _, part := range message.Content {
		switch part.Kind {
		case llmux.ContentText, llmux.ContentCommentary, llmux.ContentFinalAnswer, llmux.ContentReasoning:
			blockType := "text"
			if message.Role == llmux.RoleUser {
				blockType = "text"
			}
			content = append(content, map[string]any{"type": blockType, "text": part.Text})
		case llmux.ContentImage:
			imageURL := part.URL
			if imageURL == "" && len(part.Data) > 0 {
				mediaType := first(part.MediaType, "image/png")
				imageURL = "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
			}
			if imageURL == "" {
				return nil, fmt.Errorf("openai: image part has neither URL nor data")
			}
			content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}})
		case llmux.ContentAudio:
			if len(part.Data) == 0 {
				return nil, fmt.Errorf("openai: audio part has no data")
			}
			format := strings.TrimPrefix(part.MediaType, "audio/")
			if format == "mpeg" {
				format = "mp3"
			}
			content = append(content, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": base64.StdEncoding.EncodeToString(part.Data), "format": format}})
		case llmux.ContentToolCall:
			if part.ToolCall == nil {
				return nil, fmt.Errorf("openai: nil tool call")
			}
			arguments := part.ToolCall.Arguments
			if len(arguments) == 0 {
				arguments = json.RawMessage(`{}`)
			}
			if !json.Valid(arguments) {
				return nil, fmt.Errorf("openai: tool call %q has invalid arguments", part.ToolCall.ID)
			}
			toolCalls = append(toolCalls, map[string]any{"id": part.ToolCall.ID, "type": "function", "function": map[string]any{"name": part.ToolCall.Name, "arguments": string(arguments)}})
		default:
			return nil, fmt.Errorf("openai: content kind %q is not supported by chat completions", part.Kind)
		}
	}
	if len(content) == 0 {
		wire["content"] = nil
	} else if len(content) == 1 {
		if block, ok := content[0].(map[string]any); ok && block["type"] == "text" {
			wire["content"] = block["text"]
		} else {
			wire["content"] = content
		}
	} else {
		wire["content"] = content
	}
	if len(toolCalls) > 0 {
		wire["tool_calls"] = toolCalls
	}
	return wire, nil
}

func (model *model) responsesBody(request llmux.Request, streaming bool) (map[string]any, error) {
	input := make([]any, 0, len(request.Messages))
	instructions := request.Instructions
	for _, message := range request.Messages {
		if message.Role == llmux.RoleSystem && len(message.ProviderState) == 0 {
			text, onlyText := messageText(message)
			if onlyText {
				if instructions != "" {
					instructions += "\n\n"
				}
				instructions += text
				continue
			}
		}
		if message.Role == llmux.RoleAssistant && len(message.ProviderState) > 0 {
			var items []any
			if err := json.Unmarshal(message.ProviderState, &items); err != nil {
				return nil, fmt.Errorf("openai: invalid assistant provider state: %w", err)
			}
			input = append(input, items...)
			continue
		}
		items, err := responsesItems(message)
		if err != nil {
			return nil, err
		}
		input = append(input, items...)
	}
	body := map[string]any{"model": model.id, "input": input, "stream": streaming, "store": false}
	if instructions != "" {
		body["instructions"] = instructions
	}
	applyPortableOptions(body, request.Options, true, *model.provider.config.Profile, model.provider.Name(), model.id)
	if len(request.Metadata) > 0 {
		body["metadata"] = request.Metadata
	}
	return body, nil
}

func responsesItems(message llmux.Message) ([]any, error) {
	if message.Role == llmux.RoleTool {
		items := make([]any, 0, len(message.Content))
		for _, part := range message.Content {
			if part.Kind != llmux.ContentToolResult || part.ToolResult == nil {
				continue
			}
			output := part.ToolResult.Content
			if output == "" && len(part.ToolResult.Structured) > 0 {
				output = string(part.ToolResult.Structured)
			}
			items = append(items, map[string]any{"type": "function_call_output", "call_id": part.ToolResult.ToolCallID, "output": output})
		}
		if len(items) == 0 {
			return nil, fmt.Errorf("openai: tool message has no tool result")
		}
		return items, nil
	}
	if message.Role == llmux.RoleAssistant {
		items := make([]any, 0, len(message.Content))
		text := ""
		for _, part := range message.Content {
			switch part.Kind {
			case llmux.ContentText, llmux.ContentCommentary, llmux.ContentFinalAnswer:
				text += part.Text
			case llmux.ContentToolCall:
				if part.ToolCall == nil || !json.Valid(part.ToolCall.Arguments) {
					return nil, fmt.Errorf("openai: invalid assistant tool call")
				}
				items = append(items, map[string]any{"type": "function_call", "call_id": part.ToolCall.ID, "name": part.ToolCall.Name, "arguments": string(part.ToolCall.Arguments)})
			}
		}
		if text != "" {
			items = append([]any{map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": text}}}}, items...)
		}
		return items, nil
	}
	blocks := make([]any, 0, len(message.Content))
	for _, part := range message.Content {
		switch part.Kind {
		case llmux.ContentText, llmux.ContentCommentary, llmux.ContentFinalAnswer:
			blocks = append(blocks, map[string]any{"type": "input_text", "text": part.Text})
		case llmux.ContentImage:
			imageURL := part.URL
			if imageURL == "" && len(part.Data) > 0 {
				imageURL = "data:" + first(part.MediaType, "image/png") + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
			}
			if imageURL == "" {
				return nil, fmt.Errorf("openai: image part has neither URL nor data")
			}
			blocks = append(blocks, map[string]any{"type": "input_image", "image_url": imageURL})
		case llmux.ContentFile:
			block := map[string]any{"type": "input_file"}
			if part.URL != "" {
				block["file_url"] = part.URL
			} else if len(part.Data) > 0 {
				block["file_data"] = "data:" + first(part.MediaType, "application/octet-stream") + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
			} else {
				return nil, fmt.Errorf("openai: file part has neither URL nor data")
			}
			if part.Filename != "" {
				block["filename"] = part.Filename
			}
			blocks = append(blocks, block)
		default:
			return nil, fmt.Errorf("openai: content kind %q is not supported by Responses", part.Kind)
		}
	}
	return []any{map[string]any{"role": string(message.Role), "content": blocks}}, nil
}

func messageText(message llmux.Message) (string, bool) {
	var text strings.Builder
	for _, part := range message.Content {
		if part.Kind != llmux.ContentText {
			return "", false
		}
		text.WriteString(part.Text)
	}
	return text.String(), true
}

func applyPortableOptions(body map[string]any, options llmux.CallOptions, responses bool, profile CompatProfile, provider, modelID string) {
	if options.MaxOutputTokens != nil {
		if responses {
			body["max_output_tokens"] = *options.MaxOutputTokens
		} else if profile.UsesMaxCompletionTokens {
			body["max_completion_tokens"] = *options.MaxOutputTokens
		} else {
			body["max_tokens"] = *options.MaxOutputTokens
		}
	}
	if options.Temperature != nil {
		body["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		body["top_p"] = *options.TopP
	}
	if options.TopK != nil && profile.SupportsTopK {
		body["top_k"] = *options.TopK
	}
	if len(options.StopSequences) > 0 && profile.SupportsStop {
		body["stop"] = options.StopSequences
	}
	if options.PresencePenalty != nil && profile.SupportsPenalties {
		body["presence_penalty"] = *options.PresencePenalty
	}
	if options.FrequencyPenalty != nil && profile.SupportsPenalties {
		body["frequency_penalty"] = *options.FrequencyPenalty
	}
	if options.Seed != nil {
		body["seed"] = *options.Seed
	}
	if len(options.Tools) > 0 && profile.SupportsTools {
		tools := make([]any, 0, len(options.Tools))
		for _, tool := range options.Tools {
			if responses {
				tools = append(tools, map[string]any{"type": "function", "name": tool.Name, "description": tool.Description, "parameters": rawOrObject(tool.InputSchema), "strict": tool.Strict})
			} else {
				tools = append(tools, map[string]any{"type": "function", "function": map[string]any{"name": tool.Name, "description": tool.Description, "parameters": rawOrObject(tool.InputSchema), "strict": tool.Strict}})
			}
		}
		body["tools"] = tools
	}
	if options.ToolChoice != nil && profile.SupportsTools {
		if responses {
			if options.ToolChoice.Mode == llmux.ToolChoiceNamed {
				body["tool_choice"] = map[string]any{"type": "function", "name": options.ToolChoice.Name}
			} else {
				body["tool_choice"] = string(options.ToolChoice.Mode)
			}
		} else if options.ToolChoice.Mode == llmux.ToolChoiceNamed {
			body["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": options.ToolChoice.Name}}
		} else {
			choice := string(options.ToolChoice.Mode)
			if options.ToolChoice.Mode == llmux.ToolChoiceRequired {
				choice = profile.RequiredToolChoice
			}
			body["tool_choice"] = choice
		}
	}
	if options.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *options.ParallelToolCalls
	}
	if options.Reasoning != nil {
		reasoning := map[string]any{}
		if options.Reasoning.Effort != "" {
			reasoning["effort"] = options.Reasoning.Effort
		}
		if options.Reasoning.Summary != "" {
			reasoning["summary"] = options.Reasoning.Summary
		}
		if responses {
			body["reasoning"] = reasoning
		} else if effort := compatibleReasoningEffort(provider, modelID, options.Reasoning.Effort); effort != "" {
			body["reasoning_effort"] = effort
		}
	}
	if options.ResponseFormat != nil && profile.SupportsResponseFormat {
		format := map[string]any{"type": options.ResponseFormat.Type}
		if options.ResponseFormat.Name != "" {
			format["name"] = options.ResponseFormat.Name
		}
		if len(options.ResponseFormat.Schema) > 0 {
			format["schema"] = rawOrObject(options.ResponseFormat.Schema)
		}
		if options.ResponseFormat.Description != "" {
			format["description"] = options.ResponseFormat.Description
		}
		format["strict"] = options.ResponseFormat.Strict
		if responses {
			body["text"] = map[string]any{"format": format}
		} else {
			body["response_format"] = format
		}
	}
	if profile.DeepSeek {
		applyDeepSeek(body, options)
	}
}

func compatibleReasoningEffort(provider, modelID, effort string) string {
	if provider == "xai" {
		if xaiRejectsReasoningEffort(modelID) {
			return ""
		}
		switch effort {
		case "minimal":
			return "low"
		case "xhigh":
			return "high"
		default:
			return effort
		}
	}
	if provider != "groq" {
		return effort
	}
	switch effort {
	case "none":
		return ""
	case "minimal":
		return "low"
	case "xhigh":
		return "high"
	default:
		return effort
	}
}

func xaiRejectsReasoningEffort(modelID string) bool {
	rest, ok := strings.CutPrefix(modelID, "grok-4.20")
	if !ok {
		return false
	}
	if len(rest) >= 5 && rest[0] == '-' {
		year := rest[1:5]
		if year[0] >= '0' && year[0] <= '9' && year[1] >= '0' && year[1] <= '9' && year[2] >= '0' && year[2] <= '9' && year[3] >= '0' && year[3] <= '9' {
			rest = rest[5:]
		}
	}
	return rest == "-reasoning" || rest == "-non-reasoning"
}

func normalizeXAIOptions(options map[string]any) map[string]any {
	if len(options) == 0 {
		return options
	}
	result := make(map[string]any)
	for key, value := range options {
		switch key {
		case "reasoningEffort":
			result["reasoning_effort"] = value
		case "topLogprobs":
			result["top_logprobs"] = value
			result["logprobs"] = true
		case "searchParameters":
			result["search_parameters"] = snakeXAIObject(value)
		default:
			result[key] = value
		}
	}
	return result
}

func snakeXAIObject(value any) any {
	object, ok := value.(map[string]any)
	if !ok {
		return value
	}
	result := make(map[string]any, len(object))
	for key, item := range object {
		mapped := map[string]string{"returnCitations": "return_citations", "fromDate": "from_date", "toDate": "to_date", "maxSearchResults": "max_search_results", "excludedWebsites": "excluded_websites", "allowedWebsites": "allowed_websites", "safeSearch": "safe_search", "xHandles": "x_handles"}[key]
		if mapped == "" {
			mapped = key
		}
		if list, ok := item.([]any); ok && key == "sources" {
			converted := make([]any, len(list))
			for index, source := range list {
				converted[index] = snakeXAIObject(source)
			}
			item = converted
		}
		result[mapped] = item
	}
	return result
}

func applyDeepSeek(body map[string]any, options llmux.CallOptions) {
	deepSeek, _ := rawObject(options.ProviderOptions["deepseek"], "deepseek provider options")
	if thinking, ok := deepSeek["thinking"]; ok {
		body["thinking"] = thinking
	} else if options.Reasoning != nil && options.Reasoning.Effort != "" {
		typeName := "enabled"
		if options.Reasoning.Effort == "none" {
			typeName = "disabled"
		}
		body["thinking"] = map[string]any{"type": typeName}
	}
	delete(body, "reasoning_effort")
	if effort, ok := deepSeek["reasoningEffort"]; ok {
		body["reasoning_effort"] = effort
	} else if options.Reasoning != nil {
		effort := options.Reasoning.Effort
		switch effort {
		case "minimal":
			effort = "low"
		case "xhigh":
			effort = "max"
		case "none", "":
			return
		}
		body["reasoning_effort"] = effort
	}
}

func rawOrObject(raw json.RawMessage) any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var value any
	if json.Unmarshal(raw, &value) == nil {
		return value
	}
	return map[string]any{}
}

type chatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Choices []struct {
		Message struct {
			Content          json.RawMessage `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			ToolCalls        []chatToolCall  `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage     chatUsage       `json:"usage"`
	Citations []string        `json:"citations"`
	Code      string          `json:"code"`
	Error     json.RawMessage `json:"error"`
}

type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	PromptDetails    struct {
		CachedTokens *int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	PromptDetail struct {
		CachedTokens *int `json:"cached_tokens"`
	} `json:"prompt_token_details"`
	NumCachedTokens   *int `json:"num_cached_tokens"`
	CompletionDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func parseChatResult(payload []byte) (llmux.Result, error) {
	var response chatResponse
	if err := json.Unmarshal(payload, &response); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid chat completion: %w", err)
	}
	if len(response.Choices) == 0 {
		return llmux.Result{}, fmt.Errorf("chat completion has no choices")
	}
	choice := response.Choices[0]
	if len(response.Error) > 0 && string(response.Error) != "null" {
		var message string
		if json.Unmarshal(response.Error, &message) != nil {
			message = string(response.Error)
		}
		return llmux.Result{}, fmt.Errorf("%s: %s", response.Code, message)
	}
	text, reasoningFromContent, err := decodeChatContent(choice.Message.Content)
	if err != nil {
		return llmux.Result{}, err
	}
	result := llmux.Result{
		Text: text, Reasoning: choice.Message.ReasoningContent + reasoningFromContent, FinishReason: finishReason(choice.FinishReason), RawFinishReason: choice.FinishReason,
		Usage: usageFromChat(response.Usage), Response: llmux.ResponseMetadata{ID: response.ID, ModelID: response.Model},
	}
	if response.Created > 0 {
		result.Response.Timestamp = time.Unix(response.Created, 0)
	}
	if text != "" {
		result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentText, Text: text})
	}
	if result.Reasoning != "" {
		result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentReasoning, Text: result.Reasoning})
	}
	for _, tool := range choice.Message.ToolCalls {
		arguments := json.RawMessage(tool.Function.Arguments)
		if !json.Valid(arguments) {
			return llmux.Result{}, fmt.Errorf("tool call %q has invalid JSON arguments", tool.ID)
		}
		call := llmux.ToolCall{ID: tool.ID, Name: tool.Function.Name, Arguments: arguments}
		result.ToolCalls = append(result.ToolCalls, call)
		result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
	}
	for index, uri := range response.Citations {
		source := &llmux.Source{ID: fmt.Sprintf("citation-%d", index), URL: uri}
		result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentSource, Source: source})
	}
	return result, nil
}

func decodeChatContent(raw json.RawMessage) (string, string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", "", nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text, "", nil
	}
	var blocks []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		Thinking json.RawMessage `json:"thinking"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", "", fmt.Errorf("invalid message content: %w", err)
	}
	var builder, reasoning strings.Builder
	for _, block := range blocks {
		if block.Type == "text" || block.Type == "output_text" {
			builder.WriteString(block.Text)
		} else if block.Type == "thinking" {
			reasoning.WriteString(textFromJSON(block.Thinking))
		}
	}
	return builder.String(), reasoning.String(), nil
}

func textFromJSON(raw json.RawMessage) string {
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return ""
	}
	var walk func(any) string
	walk = func(current any) string {
		switch typed := current.(type) {
		case string:
			return typed
		case []any:
			var result strings.Builder
			for _, item := range typed {
				result.WriteString(walk(item))
			}
			return result.String()
		case map[string]any:
			return walk(typed["text"])
		default:
			return ""
		}
	}
	return walk(value)
}

type responsesResponse struct {
	ID         string            `json:"id"`
	Model      string            `json:"model"`
	CreatedAt  int64             `json:"created_at"`
	Status     string            `json:"status"`
	Output     []json.RawMessage `json:"output"`
	Usage      responsesUsage    `json:"usage"`
	Incomplete *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
}

type responsesUsage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	InputDetails     struct {
		CachedTokens     *int `json:"cached_tokens"`
		CacheWriteTokens *int `json:"cache_write_tokens"`
	} `json:"input_tokens_details"`
	OutputDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func mapResponsesUsage(usage responsesUsage) llmux.Usage {
	inputTokens := usage.InputTokens
	if inputTokens == 0 {
		inputTokens = usage.PromptTokens
	}
	outputTokens := usage.OutputTokens
	if outputTokens == 0 {
		outputTokens = usage.CompletionTokens
	}
	totalTokens := usage.TotalTokens
	if totalTokens == 0 {
		totalTokens = inputTokens + outputTokens
	}
	result := llmux.Usage{
		InputTokens: inputTokens, OutputTokens: outputTokens,
		ReasoningTokens: usage.OutputDetails.ReasoningTokens, TotalTokens: totalTokens,
	}
	if usage.InputDetails.CachedTokens != nil {
		result.CachedInputTokens = *usage.InputDetails.CachedTokens
		result.CachedInputTokensReported = true
	}
	if usage.InputDetails.CacheWriteTokens != nil {
		result.CacheWriteInputTokens = *usage.InputDetails.CacheWriteTokens
		result.CacheWriteInputTokensReported = true
	}
	return llmux.NormalizeUsage(result)
}

func parseResponsesResult(payload []byte) (llmux.Result, error) {
	var response responsesResponse
	if err := json.Unmarshal(payload, &response); err != nil {
		return llmux.Result{}, fmt.Errorf("invalid Responses payload: %w", err)
	}
	result := llmux.Result{Response: llmux.ResponseMetadata{ID: response.ID, ModelID: response.Model}, ProviderState: mustMarshal(response.Output)}
	if response.CreatedAt > 0 {
		result.Response.Timestamp = time.Unix(response.CreatedAt, 0)
	}
	result.Usage = mapResponsesUsage(response.Usage)
	for _, raw := range response.Output {
		var item struct {
			Type      string          `json:"type"`
			ID        string          `json:"id"`
			CallID    string          `json:"call_id"`
			Name      string          `json:"name"`
			Input     *string         `json:"input"`
			Arguments string          `json:"arguments"`
			Phase     llmux.TextPhase `json:"phase"`
			Summary   []struct {
				Text string `json:"text"`
			} `json:"summary"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return llmux.Result{}, fmt.Errorf("invalid Responses output item: %w", err)
		}
		switch item.Type {
		case "message":
			kind := llmux.ContentFinalAnswer
			if item.Phase == llmux.TextPhaseCommentary {
				kind = llmux.ContentCommentary
			}
			for _, block := range item.Content {
				if block.Type == "output_text" || block.Type == "refusal" {
					result.Text += block.Text
					result.Content = append(result.Content, llmux.ContentPart{Kind: kind, Text: block.Text})
				}
			}
		case "reasoning":
			for _, block := range item.Summary {
				result.Reasoning += block.Text
				result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentReasoning, Text: block.Text})
			}
		case "function_call":
			arguments := json.RawMessage(item.Arguments)
			if !json.Valid(arguments) {
				return llmux.Result{}, fmt.Errorf("tool call %q has invalid JSON arguments", item.CallID)
			}
			call := llmux.ToolCall{ID: first(item.CallID, item.ID), Name: item.Name, Arguments: arguments}
			result.ToolCalls = append(result.ToolCalls, call)
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
		case "custom_tool_call":
			input := ""
			if item.Input != nil {
				input = *item.Input
			}
			call := llmux.ToolCall{ID: first(item.CallID, item.ID), Name: item.Name, Arguments: customToolArguments(input)}
			result.ToolCalls = append(result.ToolCalls, call)
			result.Content = append(result.Content, llmux.ContentPart{Kind: llmux.ContentToolCall, ToolCall: &call})
		}
	}
	result.FinishReason, result.RawFinishReason = responsesFinish(response)
	return result, nil
}

func responsesFinish(response responsesResponse) (llmux.FinishReason, string) {
	if response.Status == "completed" {
		return llmux.FinishStop, response.Status
	}
	if response.Incomplete != nil {
		if response.Incomplete.Reason == "max_output_tokens" {
			return llmux.FinishLength, response.Incomplete.Reason
		}
		return llmux.FinishError, response.Incomplete.Reason
	}
	return llmux.FinishUnknown, response.Status
}

func finishReason(raw string) llmux.FinishReason {
	switch raw {
	case "stop":
		return llmux.FinishStop
	case "length", "max_tokens", "model_length":
		return llmux.FinishLength
	case "tool_calls", "function_call":
		return llmux.FinishToolCalls
	case "content_filter":
		return llmux.FinishContent
	case "error":
		return llmux.FinishError
	default:
		return llmux.FinishUnknown
	}
}

func usageFromChat(usage chatUsage) llmux.Usage {
	cached, reported := maximumReported(usage.PromptDetails.CachedTokens, usage.PromptDetail.CachedTokens, usage.NumCachedTokens)
	return llmux.NormalizeUsage(llmux.Usage{
		InputTokens:               usage.PromptTokens,
		CachedInputTokens:         cached,
		CachedInputTokensReported: reported,
		OutputTokens:              usage.CompletionTokens,
		ReasoningTokens:           usage.CompletionDetails.ReasoningTokens,
		TotalTokens:               usage.TotalTokens,
	})
}

func maximumReported(values ...*int) (int, bool) {
	maximum := 0
	reported := false
	for _, value := range values {
		if value == nil {
			continue
		}
		reported = true
		maximum = max(maximum, *value)
	}
	return maximum, reported
}

func mustMarshal(value any) json.RawMessage {
	result, _ := json.Marshal(value)
	return result
}
