package llmux

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

type capabilityProvider struct{}

func (capabilityProvider) Name() string { return "capability-test" }
func (capabilityProvider) LanguageModel(string) (LanguageModel, error) {
	return nil, errors.New("unused")
}

func (capabilityProvider) Descriptor() ProviderDescriptor {
	return ProviderDescriptor{Name: "capability-test", WireProtocols: []string{"test-wire"}}
}

func (capabilityProvider) EmbeddingModel(string) (EmbeddingModel, error) {
	return capabilityEmbedding{}, nil
}

type capabilityEmbedding struct{}

func (capabilityEmbedding) ModelID() string { return "embed" }
func (capabilityEmbedding) Embed(context.Context, []string, EmbeddingOptions) (EmbeddingResult, error) {
	return EmbeddingResult{}, nil
}

func TestDescribeProviderCombinesExplicitAndOptionalCapabilities(t *testing.T) {
	descriptor, err := DescribeProvider(capabilityProvider{})
	if err != nil {
		t.Fatal(err)
	}
	if descriptor.Name != "capability-test" || len(descriptor.WireProtocols) != 1 {
		t.Fatalf("descriptor = %#v", descriptor)
	}
	want := []ProviderCapability{CapabilityLanguage, CapabilityEmbedding}
	if len(descriptor.Capabilities) != len(want) {
		t.Fatalf("capabilities = %#v", descriptor.Capabilities)
	}
	for index := range want {
		if descriptor.Capabilities[index] != want[index] {
			t.Fatalf("capabilities = %#v", descriptor.Capabilities)
		}
	}
	model, err := OpenEmbeddingModel(capabilityProvider{}, "embed")
	if err != nil || model.ModelID() != "embed" {
		t.Fatalf("embedding factory = %#v, %v", model, err)
	}
	if _, err := OpenFiles(capabilityProvider{}); err != nil {
		var providerErr *ProviderError
		if !errors.As(err, &providerErr) || providerErr.Kind != ErrorUnsupported {
			t.Fatalf("unsupported files error = %v", err)
		}
	}
}

func TestModelCapabilitiesJSONRoundTrip(t *testing.T) {
	yes := true
	input := ModelInfo{
		ID: "model",
		Capabilities: &ModelCapabilities{
			InputModalities: []Modality{ModalityText, ModalityImage},
			ContextWindow:   128_000,
			ToolCalling:     &yes,
		},
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ModelInfo
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Capabilities == nil || decoded.Capabilities.ContextWindow != 128_000 ||
		decoded.Capabilities.ToolCalling == nil || !*decoded.Capabilities.ToolCalling {
		t.Fatalf("decoded capabilities = %#v", decoded.Capabilities)
	}
	legacy, err := json.Marshal(ModelInfo{ID: "legacy"})
	if err != nil || string(legacy) != `{"id":"legacy"}` {
		t.Fatalf("legacy model JSON = %s, %v", legacy, err)
	}
}
