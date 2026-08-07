package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

// ListModels implements llmux.ModelLister using Anthropic's Models API:
// GET /v1/models (paginated).
func (provider *Provider) ListModels(ctx context.Context) ([]llmux.ModelInfo, error) {
	const pageLimit = 100
	result := make([]llmux.ModelInfo, 0, pageLimit)
	afterID := ""
	for {
		endpoint := provider.config.BaseURL + "/v1/models?limit=" + strconv.Itoa(pageLimit)
		if afterID != "" {
			endpoint += "&after_id=" + url.QueryEscape(afterID)
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
		payload, status, header := readListBody(response)
		_ = response.Body.Close()
		if status/100 != 2 {
			return nil, provider.listResponseError(status, header, payload)
		}
		page, hasMore, lastID, err := parseAnthropicModelPage(payload)
		if err != nil {
			return nil, err
		}
		result = append(result, page...)
		if !hasMore || lastID == "" || lastID == afterID {
			break
		}
		afterID = lastID
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
		headers.Set(provider.config.APIKeyHeader, provider.config.APIKeyPrefix+provider.config.APIKey)
	}
	headers.Set("Anthropic-Version", provider.config.Version)
	if len(provider.config.Beta) > 0 {
		headers.Set("Anthropic-Beta", strings.Join(provider.config.Beta, ","))
	}
	return headers
}

func parseAnthropicModelPage(payload []byte) ([]llmux.ModelInfo, bool, string, error) {
	var envelope struct {
		Data []json.RawMessage `json:"data"`
		// has_more may be bool or omitted.
		HasMore *bool  `json:"has_more"`
		LastID  string `json:"last_id"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return nil, false, "", fmt.Errorf("anthropic: invalid models response: %w", err)
	}
	result := make([]llmux.ModelInfo, 0, len(envelope.Data))
	var lastID string
	for _, raw := range envelope.Data {
		var item struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
			CreatedAt   string `json:"created_at"`
			Type        string `json:"type"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, false, "", fmt.Errorf("anthropic: invalid model entry: %w", err)
		}
		if strings.TrimSpace(item.ID) == "" {
			continue
		}
		info := llmux.ModelInfo{
			ID:          item.ID,
			DisplayName: item.DisplayName,
			Raw:         append(json.RawMessage(nil), raw...),
		}
		if item.CreatedAt != "" {
			if when, err := time.Parse(time.RFC3339, item.CreatedAt); err == nil {
				info.Created = when.Unix()
			}
		}
		result = append(result, info)
		lastID = item.ID
	}
	if envelope.LastID != "" {
		lastID = envelope.LastID
	}
	hasMore := false
	if envelope.HasMore != nil {
		hasMore = *envelope.HasMore
	} else if len(envelope.Data) > 0 && lastID != "" {
		// Older gateways may omit has_more; stop after one page unless full.
		hasMore = len(envelope.Data) >= pageLimitHint
	}
	return result, hasMore, lastID, nil
}

const pageLimitHint = 100

func readListBody(response *http.Response) ([]byte, int, http.Header) {
	payload, _ := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	return payload, response.StatusCode, response.Header
}

func (provider *Provider) listResponseError(status int, header http.Header, payload []byte) error {
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := firstNonEmpty(envelope.Error.Message, envelope.Message, strings.TrimSpace(string(payload)))
	code := firstNonEmpty(envelope.Error.Type, envelope.Type)
	providerError := &llmux.ProviderError{
		Provider: provider.Name(), Kind: llmux.ErrorKindForStatus(status), Code: code,
		StatusCode: status, Message: boundMessage(message, 8<<10),
	}
	if retry := header.Get("Retry-After"); retry != "" {
		if duration, err := time.ParseDuration(retry + "s"); err == nil {
			providerError.RetryAfter = duration
		}
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

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func boundMessage(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "…"
}

var _ llmux.ModelLister = (*Provider)(nil)