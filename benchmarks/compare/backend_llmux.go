package main

import (
	"context"
	"errors"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/openai"
)

const (
	backendName   = "llmux-pure-go"
	backendCommit = "working-tree"
)

type llmuxBackend struct {
	model   llmux.LanguageModel
	request llmux.Request
}

func newBenchmarkBackend(baseURL string) (benchmarkBackend, error) {
	provider, err := openai.New(openai.Config{APIKey: "bench", BaseURL: baseURL, Retry: llmux.RetryPolicy{MaxAttempts: 1}, WireAPI: openai.ChatCompletions})
	if err != nil {
		return nil, err
	}
	model, err := provider.LanguageModel("bench")
	if err != nil {
		return nil, err
	}
	return &llmuxBackend{model: model, request: llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")}}}, nil
}

func (b *llmuxBackend) Generate() error {
	_, err := b.model.Generate(context.Background(), b.request)
	return err
}

func (b *llmuxBackend) Stream() error {
	stream, err := b.model.Stream(context.Background(), b.request)
	if err != nil {
		return err
	}
	_, err = llmux.Collect(stream)
	return err
}

func (b *llmuxBackend) CancelLatency() (time.Duration, bool, error) {
	ctx, cancel := context.WithCancel(context.Background())
	request := b.request
	request.Options.Headers = map[string]string{"X-Bench-Slow": "1"}
	stream, err := b.model.Stream(ctx, request)
	if err != nil {
		cancel()
		return 0, true, err
	}
	if _, err := stream.Recv(); err != nil {
		cancel()
		return 0, true, err
	}
	started := time.Now()
	cancel()
	for {
		_, err = stream.Recv()
		if err != nil {
			break
		}
	}
	latency := time.Since(started)
	_ = stream.Close()
	if !errors.Is(err, context.Canceled) {
		return latency, true, err
	}
	return latency, true, nil
}

func (*llmuxBackend) Close() error { return nil }
