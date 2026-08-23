package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

// ListModels implements llmux.ModelLister using the Gemini Models API:
// GET /v1beta/models (paginated).
func (provider *Provider) ListModels(ctx context.Context) ([]llmux.ModelInfo, error) {
	result := make([]llmux.ModelInfo, 0, 64)
	pageToken := ""
	for {
		endpoint := provider.config.BaseURL + "/models?pageSize=100"
		if pageToken != "" {
			endpoint += "&pageToken=" + url.QueryEscape(pageToken)
		}
		response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{
			Method:  http.MethodGet,
			URL:     endpoint,
			Headers: provider.listHeaders(),
			Retry:   provider.config.Retry,
		})
		if err != nil {
			return nil, provider.listTransportError(err)
		}
		payload, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
		status := response.StatusCode
		_ = response.Body.Close()
		if err != nil {
			return nil, provider.listTransportError(err)
		}
		if status/100 != 2 {
			return nil, provider.listResponseError(status, payload)
		}
		page, next, err := parseGoogleModelPage(payload)
		if err != nil {
			return nil, err
		}
		result = append(result, page...)
		if next == "" || next == pageToken {
			break
		}
		pageToken = next
	}
	return result, nil
}

func (provider *Provider) listHeaders() http.Header {
	headers := provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Accept", "application/json")
	if provider.config.APIKey != "" {
		headers.Set("X-Goog-Api-Key", provider.config.APIKey)
	}
	return headers
}

func parseGoogleModelPage(payload []byte) ([]llmux.ModelInfo, string, error) {
	var envelope struct {
		Models        []json.RawMessage `json:"models"`
		NextPageToken string            `json:"nextPageToken"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return nil, "", fmt.Errorf("google: invalid models response: %w", err)
	}
	result := make([]llmux.ModelInfo, 0, len(envelope.Models))
	for _, raw := range envelope.Models {
		var item struct {
			Name                       string   `json:"name"`
			DisplayName                string   `json:"displayName"`
			Description                string   `json:"description"`
			InputTokenLimit            int      `json:"inputTokenLimit"`
			OutputTokenLimit           int      `json:"outputTokenLimit"`
			SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, "", fmt.Errorf("google: invalid model entry: %w", err)
		}
		id := strings.TrimSpace(item.Name)
		id = strings.TrimPrefix(id, "models/")
		if id == "" {
			continue
		}
		capabilities := googleModelCapabilities(
			item.SupportedGenerationMethods,
			item.InputTokenLimit,
			item.OutputTokenLimit,
		)
		result = append(result, llmux.ModelInfo{
			ID:           id,
			DisplayName:  item.DisplayName,
			Description:  item.Description,
			Capabilities: capabilities,
			Raw:          append(json.RawMessage(nil), raw...),
		})
	}
	return result, envelope.NextPageToken, nil
}

func googleModelCapabilities(methods []string, inputLimit, outputLimit int) *llmux.ModelCapabilities {
	if methods == nil && inputLimit == 0 && outputLimit == 0 {
		return nil
	}
	capabilities := &llmux.ModelCapabilities{
		ContextWindow:   inputLimit,
		MaxOutputTokens: outputLimit,
	}
	if methods == nil {
		return capabilities
	}
	streaming := false
	for _, method := range methods {
		switch method {
		case "generateContent":
			capabilities.InputModalities = appendModality(capabilities.InputModalities, llmux.ModalityText)
			capabilities.OutputModalities = appendModality(capabilities.OutputModalities, llmux.ModalityText)
		case "streamGenerateContent":
			streaming = true
			capabilities.InputModalities = appendModality(capabilities.InputModalities, llmux.ModalityText)
			capabilities.OutputModalities = appendModality(capabilities.OutputModalities, llmux.ModalityText)
		case "embedContent", "batchEmbedContents":
			capabilities.InputModalities = appendModality(capabilities.InputModalities, llmux.ModalityText)
			capabilities.OutputModalities = appendModality(capabilities.OutputModalities, llmux.ModalityEmbedding)
		}
	}
	capabilities.Streaming = &streaming
	return capabilities
}

func appendModality(values []llmux.Modality, value llmux.Modality) []llmux.Modality {
	for _, current := range values {
		if current == value {
			return values
		}
	}
	return append(values, value)
}

func (provider *Provider) listResponseError(status int, payload []byte) error {
	var envelope struct {
		Error struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := envelope.Error.Message
	if message == "" {
		message = strings.TrimSpace(string(payload))
	}
	code := envelope.Error.Status
	if code == "" && envelope.Error.Code != 0 {
		code = fmt.Sprint(envelope.Error.Code)
	}
	providerError := &llmux.ProviderError{
		Provider: provider.Name(), Kind: llmux.ErrorKindForStatus(status), Code: code,
		StatusCode: status, Message: message,
	}
	if json.Valid(payload) {
		providerError.Raw = append(json.RawMessage(nil), payload...)
	}
	return providerError
}

func (provider *Provider) listTransportError(err error) error {
	kind := llmux.ErrorStream
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		kind = llmux.ErrorCancelled
	}
	return &llmux.ProviderError{Provider: provider.Name(), Kind: kind, Message: err.Error(), Cause: err}
}

var _ llmux.ModelLister = (*Provider)(nil)
