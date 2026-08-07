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

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

// ListModels implements llmux.ModelLister via Bedrock ListFoundationModels.
// The response is the regional foundation catalog, not account entitlement.
func (provider *Provider) ListModels(ctx context.Context) ([]llmux.ModelInfo, error) {
	endpoint := provider.listModelsURL()
	headers, err := provider.listHeaders(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{
		Method:  http.MethodGet,
		URL:     endpoint,
		Headers: headers,
		Retry:   provider.config.Retry,
	})
	if err != nil {
		return nil, provider.listTransportError(err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return nil, provider.listTransportError(err)
	}
	if response.StatusCode/100 != 2 {
		return nil, provider.listResponseError(response.StatusCode, payload)
	}
	return parseBedrockModelList(payload)
}

func (provider *Provider) listModelsURL() string {
	base := provider.controlPlaneBase()
	return base + "/foundation-models"
}

// controlPlaneBase maps the default runtime host to the control-plane host.
// Custom BaseURL values (tests, proxies) are used as-is.
func (provider *Provider) controlPlaneBase() string {
	base := provider.config.BaseURL
	if strings.Contains(base, "bedrock-runtime.") {
		return "https://bedrock." + provider.config.Region + ".amazonaws.com"
	}
	return base
}

func (provider *Provider) listHeaders(ctx context.Context, method, endpoint string, body []byte) (http.Header, error) {
	headers := provider.config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	headers.Set("Accept", "application/json")
	if provider.config.BearerToken != "" {
		headers.Set("Authorization", "Bearer "+provider.config.BearerToken)
		return headers, nil
	}
	credentials := provider.config.Credentials
	if provider.config.CredentialsProvider != nil {
		var err error
		credentials, err = provider.config.CredentialsProvider(ctx)
		if err != nil {
			return nil, &llmux.ProviderError{Provider: "amazon-bedrock", Kind: llmux.ErrorAuthentication, Message: err.Error(), Cause: err}
		}
	}
	if credentials.AccessKeyID == "" || credentials.SecretAccessKey == "" {
		return nil, errors.New("bedrock: credentials provider returned empty credentials")
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("bedrock: invalid list models URL: %w", err)
	}
	signV4(method, parsed, body, headers, credentials, provider.config.Region, provider.config.Now())
	return headers, nil
}

func parseBedrockModelList(payload []byte) ([]llmux.ModelInfo, error) {
	var envelope struct {
		ModelSummaries []json.RawMessage `json:"modelSummaries"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return nil, fmt.Errorf("bedrock: invalid models response: %w", err)
	}
	result := make([]llmux.ModelInfo, 0, len(envelope.ModelSummaries))
	for _, raw := range envelope.ModelSummaries {
		var item struct {
			ModelID   string `json:"modelId"`
			ModelName string `json:"modelName"`
			Provider  string `json:"providerName"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, fmt.Errorf("bedrock: invalid model entry: %w", err)
		}
		if strings.TrimSpace(item.ModelID) == "" {
			continue
		}
		result = append(result, llmux.ModelInfo{
			ID:          item.ModelID,
			DisplayName: item.ModelName,
			OwnedBy:     item.Provider,
			Raw:         append(json.RawMessage(nil), raw...),
		})
	}
	return result, nil
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
		Provider: "amazon-bedrock", Kind: llmux.ErrorKindForStatus(status),
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
	return &llmux.ProviderError{Provider: "amazon-bedrock", Kind: kind, Message: err.Error(), Cause: err}
}

var _ llmux.ModelLister = (*Provider)(nil)