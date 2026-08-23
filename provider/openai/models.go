package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

// ListModels implements llmux.ModelLister using the OpenAI Models protocol
// family: GET {base}/models (or Config.ListModelsURL when set).
func (provider *Provider) ListModels(ctx context.Context) ([]llmux.ModelInfo, error) {
	response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{
		Method:  http.MethodGet,
		URL:     provider.listModelsURL(),
		Headers: provider.listHeaders(),
		Retry:   provider.config.Retry,
	})
	if err != nil {
		return nil, provider.listTransportError(err)
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return nil, provider.listResponseError(response)
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return nil, provider.listTransportError(err)
	}
	return parseOpenAIModelList(payload)
}

func (provider *Provider) listModelsURL() string {
	if provider.config.ListModelsURL != "" {
		return provider.config.ListModelsURL
	}
	return provider.config.BaseURL + "/models"
}

func (provider *Provider) listHeaders() http.Header {
	headers := provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Accept", "application/json")
	if provider.config.APIKey != "" {
		headers.Set(provider.config.APIKeyHeader, provider.config.APIKeyPrefix+provider.config.APIKey)
	}
	if provider.config.Organization != "" {
		headers.Set("OpenAI-Organization", provider.config.Organization)
	}
	if provider.config.Project != "" {
		headers.Set("OpenAI-Project", provider.config.Project)
	}
	return headers
}

func parseOpenAIModelList(payload []byte) ([]llmux.ModelInfo, error) {
	var envelope struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return nil, fmt.Errorf("openai: invalid models response: %w", err)
	}
	if envelope.Data == nil {
		// Some gateways return a bare array.
		var bare []json.RawMessage
		if err := json.Unmarshal(payload, &bare); err == nil {
			envelope.Data = bare
		}
	}
	result := make([]llmux.ModelInfo, 0, len(envelope.Data))
	for _, raw := range envelope.Data {
		var item struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
			Created int64  `json:"created"`
			// Azure-style alternate timestamp.
			CreatedAt int64 `json:"created_at"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, fmt.Errorf("openai: invalid model entry: %w", err)
		}
		if strings.TrimSpace(item.ID) == "" {
			continue
		}
		created := item.Created
		if created == 0 {
			created = item.CreatedAt
		}
		result = append(result, llmux.ModelInfo{
			ID:      item.ID,
			OwnedBy: item.OwnedBy,
			Created: created,
			Raw:     append(json.RawMessage(nil), raw...),
		})
	}
	return result, nil
}

func (provider *Provider) listResponseError(response *http.Response) error {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
		Message string `json:"message"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := first(envelope.Error.Message, envelope.Message, strings.TrimSpace(string(payload)))
	code := envelope.Error.Type
	if envelope.Error.Code != nil {
		code = fmt.Sprint(envelope.Error.Code)
	}
	providerError := &llmux.ProviderError{
		Provider: provider.Name(), Kind: llmux.ErrorKindForStatus(response.StatusCode), Code: code,
		StatusCode: response.StatusCode, Message: bounded(message, 8<<10), RetryAfter: retryAfter(response.Header.Get("Retry-After")),
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
