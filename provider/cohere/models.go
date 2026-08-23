package cohere

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

// ListModels implements llmux.ModelLister using Cohere's Models API:
// GET /v1/models (paginated). Chat uses /v2; discovery remains on /v1.
func (provider *Provider) ListModels(ctx context.Context) ([]llmux.ModelInfo, error) {
	result := make([]llmux.ModelInfo, 0, 32)
	pageToken := ""
	for {
		endpoint := provider.listModelsURL()
		if pageToken != "" {
			if strings.Contains(endpoint, "?") {
				endpoint += "&page_token=" + url.QueryEscape(pageToken)
			} else {
				endpoint += "?page_token=" + url.QueryEscape(pageToken)
			}
		} else if !strings.Contains(endpoint, "?") {
			endpoint += "?page_size=100"
		} else {
			endpoint += "&page_size=100"
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
		page, next, err := parseCohereModelPage(payload)
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

func (provider *Provider) listModelsURL() string {
	base := provider.config.BaseURL
	switch {
	case strings.HasSuffix(base, "/v2"):
		return strings.TrimSuffix(base, "/v2") + "/v1/models"
	case strings.HasSuffix(base, "/v1"):
		return base + "/models"
	default:
		return base + "/v1/models"
	}
}

func (provider *Provider) listHeaders() http.Header {
	headers := provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Accept", "application/json")
	headers.Set("Authorization", "Bearer "+provider.config.APIKey)
	return headers
}

func parseCohereModelPage(payload []byte) ([]llmux.ModelInfo, string, error) {
	var envelope struct {
		Models        []json.RawMessage `json:"models"`
		NextPageToken string            `json:"next_page_token"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return nil, "", fmt.Errorf("cohere: invalid models response: %w", err)
	}
	result := make([]llmux.ModelInfo, 0, len(envelope.Models))
	for _, raw := range envelope.Models {
		var item struct {
			Name    string `json:"name"`
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, "", fmt.Errorf("cohere: invalid model entry: %w", err)
		}
		id := strings.TrimSpace(item.Name)
		if id == "" {
			id = strings.TrimSpace(item.ID)
		}
		if id == "" {
			continue
		}
		result = append(result, llmux.ModelInfo{
			ID:      id,
			OwnedBy: item.OwnedBy,
			Raw:     append(json.RawMessage(nil), raw...),
		})
	}
	return result, envelope.NextPageToken, nil
}

func (provider *Provider) listResponseError(status int, payload []byte) error {
	var envelope struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := envelope.Message
	if message == "" {
		message = strings.TrimSpace(string(payload))
	}
	providerError := &llmux.ProviderError{
		Provider: "cohere", Kind: llmux.ErrorKindForStatus(status),
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
	return &llmux.ProviderError{Provider: "cohere", Kind: kind, Message: err.Error(), Cause: err}
}

var _ llmux.ModelLister = (*Provider)(nil)
