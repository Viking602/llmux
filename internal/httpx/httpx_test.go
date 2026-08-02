package httpx

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Viking602/llmux"
)

func TestDoRetriesBeforeReturningResponse(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if calls.Add(1) < 3 {
			response.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(response, "ok")
	}))
	defer server.Close()

	response, err := Do(context.Background(), server.Client(), Request{
		Method: http.MethodPost,
		URL:    server.URL,
		Body:   []byte("request"),
		Retry:  llmux.RetryPolicy{MaxAttempts: 3, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if calls.Load() != 3 || response.StatusCode != http.StatusOK {
		t.Fatalf("calls/status = %d/%d", calls.Load(), response.StatusCode)
	}
}

func TestDoStopsWhileBackoffContextIsCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Retry-After", "30")
		response.WriteHeader(http.StatusTooManyRequests)
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := Do(ctx, server.Client(), Request{
		Method: http.MethodGet,
		URL:    server.URL,
		Retry:  llmux.RetryPolicy{MaxAttempts: 2, MaxDelay: time.Second},
	})
	if err != context.DeadlineExceeded {
		t.Fatalf("error = %v", err)
	}
}

func TestDoLeavesFinalRetryableResponseReadable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(response, "provider unavailable")
	}))
	defer server.Close()

	response, err := Do(context.Background(), server.Client(), Request{
		Method: http.MethodPost,
		URL:    server.URL,
		Retry:  llmux.RetryPolicy{MaxAttempts: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "provider unavailable" {
		t.Fatalf("body = %q", body)
	}
}
