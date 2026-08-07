package llmux_test

import (
	"context"
	"errors"
	"testing"

	"github.com/Viking602/llmux"
)

type stubProvider struct{ name string }

func (s stubProvider) Name() string { return s.name }
func (stubProvider) LanguageModel(string) (llmux.LanguageModel, error) {
	return nil, errors.New("unused")
}

type stubLister struct {
	stubProvider
	models []llmux.ModelInfo
}

func (s stubLister) ListModels(context.Context) ([]llmux.ModelInfo, error) {
	return s.models, nil
}

func TestListModelsUnsupported(t *testing.T) {
	_, err := llmux.ListModels(context.Background(), stubProvider{name: "stub"})
	var providerErr *llmux.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Kind != llmux.ErrorUnsupported {
		t.Fatalf("error = %v", err)
	}
}

func TestListModelsSupported(t *testing.T) {
	want := []llmux.ModelInfo{{ID: "m1"}}
	got, err := llmux.ListModels(context.Background(), stubLister{
		stubProvider: stubProvider{name: "stub"},
		models:       want,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "m1" {
		t.Fatalf("got = %#v", got)
	}
}
