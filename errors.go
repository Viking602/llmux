package llmux

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type ErrorKind string

const (
	ErrorUnknown        ErrorKind = "unknown"
	ErrorAuthentication ErrorKind = "authentication"
	ErrorTokenExpired   ErrorKind = "token_expired"
	ErrorPermission     ErrorKind = "permission"
	ErrorInvalidRequest ErrorKind = "invalid_request"
	ErrorNotFound       ErrorKind = "not_found"
	ErrorConflict       ErrorKind = "conflict"
	ErrorRateLimit      ErrorKind = "rate_limit"
	ErrorServer         ErrorKind = "server"
	ErrorStream         ErrorKind = "stream"
	ErrorCancelled      ErrorKind = "cancelled"
)

type ProviderError struct {
	Provider   string          `json:"provider,omitempty"`
	Kind       ErrorKind       `json:"kind"`
	Code       string          `json:"code,omitempty"`
	StatusCode int             `json:"statusCode,omitempty"`
	Message    string          `json:"message,omitempty"`
	RetryAfter time.Duration   `json:"retryAfter,omitempty"`
	Raw        json.RawMessage `json:"raw,omitempty"`
	Cause      error           `json:"-"`
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	label := e.Provider
	if e.Code != "" {
		if label != "" {
			label += " "
		}
		label += e.Code
	}
	if e.StatusCode != 0 {
		label = fmt.Sprintf("%s HTTP %d", label, e.StatusCode)
	}
	if label == "" {
		label = string(e.Kind)
	}
	if e.Message == "" {
		return label
	}
	return label + ": " + e.Message
}

func (e *ProviderError) Unwrap() error { return e.Cause }

func (e *ProviderError) Retryable() bool {
	return e != nil && (e.Kind == ErrorRateLimit || e.Kind == ErrorServer || e.Kind == ErrorStream)
}

func ErrorKindForStatus(status int) ErrorKind {
	switch {
	case status == http.StatusUnauthorized:
		return ErrorAuthentication
	case status == http.StatusForbidden:
		return ErrorPermission
	case status == http.StatusNotFound:
		return ErrorNotFound
	case status == http.StatusConflict:
		return ErrorConflict
	case status == http.StatusTooManyRequests:
		return ErrorRateLimit
	case status >= 500:
		return ErrorServer
	case status >= 400:
		return ErrorInvalidRequest
	default:
		return ErrorUnknown
	}
}
