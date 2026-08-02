package httpx

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Viking602/llmux"
)

const (
	defaultResponseHeaderTimeout = 30 * time.Second
	defaultBaseDelay             = 200 * time.Millisecond
	defaultMaxDelay              = 2 * time.Second
	maxRetryDrainBytes           = 64 << 10
)

func NewClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 100
	transport.MaxIdleConnsPerHost = 20
	transport.IdleConnTimeout = 90 * time.Second
	transport.ResponseHeaderTimeout = defaultResponseHeaderTimeout
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return errors.New("llmux: stopped after 10 redirects")
			}
			if len(via) > 0 && !sameOrigin(via[0].URL, req.URL) {
				// Never risk forwarding provider credentials to another origin.
				return http.ErrUseLastResponse
			}
			return nil
		},
	}
}

func sameOrigin(left, right *url.URL) bool {
	return left != nil && right != nil && strings.EqualFold(left.Scheme, right.Scheme) && strings.EqualFold(left.Host, right.Host)
}

type Request struct {
	Method  string
	URL     string
	Headers http.Header
	Body    []byte
	Retry   llmux.RetryPolicy
}

// Do retries only transport failures and retryable HTTP statuses. The request
// body is immutable bytes so each attempt gets a fresh reader.
func Do(ctx context.Context, client *http.Client, spec Request) (*http.Response, error) {
	if client == nil {
		client = NewClient()
	}
	policy := normalizePolicy(spec.Retry)
	var lastErr error
	for attempt := 1; attempt <= policy.MaxAttempts; attempt++ {
		req, err := http.NewRequestWithContext(ctx, spec.Method, spec.URL, bytes.NewReader(spec.Body))
		if err != nil {
			return nil, err
		}
		req.Header = spec.Headers.Clone()
		response, err := client.Do(req)
		if err == nil && !retryableStatus(response.StatusCode) {
			return response, nil
		}
		if err != nil && ctx.Err() != nil {
			if response != nil {
				_ = response.Body.Close()
			}
			return nil, ctx.Err()
		}
		lastErr = err
		var retryAfter time.Duration
		if response != nil {
			retryAfter = parseRetryAfter(response.Header.Get("Retry-After"), time.Now())
			if attempt == policy.MaxAttempts {
				return response, nil
			}
			_, _ = io.CopyN(io.Discard, response.Body, maxRetryDrainBytes)
			_ = response.Body.Close()
		}
		if attempt == policy.MaxAttempts {
			return nil, lastErr
		}
		delay := backoff(policy, attempt, retryAfter)
		if err := wait(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func normalizePolicy(policy llmux.RetryPolicy) llmux.RetryPolicy {
	if policy.MaxAttempts == 0 {
		policy.MaxAttempts = 3
	}
	if policy.MaxAttempts < 1 {
		policy.MaxAttempts = 1
	}
	if policy.MaxAttempts > 10 {
		policy.MaxAttempts = 10
	}
	if policy.BaseDelay <= 0 {
		policy.BaseDelay = defaultBaseDelay
	}
	if policy.MaxDelay <= 0 {
		policy.MaxDelay = defaultMaxDelay
	}
	if policy.MaxDelay < policy.BaseDelay {
		policy.MaxDelay = policy.BaseDelay
	}
	return policy
}

func retryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

func backoff(policy llmux.RetryPolicy, attempt int, retryAfter time.Duration) time.Duration {
	shift := min(max(attempt-1, 0), 30)
	ceiling := min(policy.BaseDelay*time.Duration(1<<shift), policy.MaxDelay)
	if retryAfter > ceiling {
		ceiling = min(retryAfter, policy.MaxDelay)
	}
	if ceiling <= 0 {
		return 0
	}
	return time.Duration(rand.Int64N(int64(ceiling) + 1))
}

func wait(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func parseRetryAfter(value string, now time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		return max(time.Duration(seconds)*time.Second, 0)
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0
	}
	return max(when.Sub(now), 0)
}
