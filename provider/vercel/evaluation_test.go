package vercel

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestEvaluationWireAndFailureBoundaries(t *testing.T) {
	input := llmux.EvaluationRequest{State: json.RawMessage(`{"diff":"removed name"}`), Questions: map[string]llmux.EvaluationQuestion{
		"contract": {Type: "choice", Instructions: "Judge compatibility", Criteria: json.RawMessage(`{"HIT":"break","CLEAR":"compatible","MISSING":"unknown"}`)},
		"verified": {Type: "boolean", Instructions: "Is evidence verified?"},
		"quality":  {Type: "score", Instructions: "Judge evidence quality", Criteria: json.RawMessage(`["missing","partial","complete"]`)},
	}}
	valid := `{"answers":{"contract":{"type":"choice","choice":"HIT","probabilities":{"HIT":0.9,"CLEAR":0.05,"MISSING":0.05}},"verified":{"type":"boolean","probability":0},"quality":{"type":"score","score":1.5}},"usage":{"inputTokens":20,"outputTokens":8}}`
	for _, tc := range []struct {
		name   string
		status int
		body   string
		ok     bool
	}{
		{"success", 200, valid, true},
		{"authentication", 401, `secret must not leak`, false},
		{"server_no_retry", 500, `server`, false},
		{"missing_answers", 200, `{"answers":{}}`, false},
		{"bad_probability", 200, `{"answers":{"contract":{"type":"choice","choice":"HIT"},"verified":{"type":"boolean","probability":2},"quality":{"type":"score","score":0}}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.Method != "POST" || r.URL.Path != "/v4/ai/evaluation-model" || r.Header.Get("Authorization") != "Bearer test-key" || r.Header.Get("ai-model-id") != "typesafe-ai/jev" || r.Header.Get("ai-evaluation-model-specification-version") != "4" || r.Header.Get("ai-gateway-auth-method") != "api-key" {
					t.Error("wrong evaluation wire request")
				}
				var body map[string]json.RawMessage
				if json.NewDecoder(r.Body).Decode(&body) != nil || body["messages"] != nil || body["model"] != nil || body["questions"] == nil || body["state"] == nil {
					t.Error("wrong evaluation body")
				}
				w.Header().Set("x-request-id", "evaluation-1")
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL + "/v4/ai"})
			if err != nil {
				t.Fatal(err)
			}
			model, _ := provider.EvaluationModel("typesafe-ai/jev")
			result, err := model.Evaluate(context.Background(), input)
			if (err == nil) != tc.ok || calls != 1 {
				t.Fatalf("result=%+v err=%v calls=%d", result, err, calls)
			}
			if tc.ok && (result.Answers["verified"].Probability == nil || *result.Answers["verified"].Probability != 0 || result.Usage.InputTokens != 20 || result.Response.ID != "evaluation-1") {
				t.Fatal("typed result lost")
			}
			if tc.status == 401 {
				var e *llmux.ProviderError
				if !errors.As(err, &e) || e.StatusCode != 401 || e.Message != "" || e.Raw != nil {
					t.Fatal("unsafe provider error")
				}
			}
			bad := input
			bad.Questions = map[string]llmux.EvaluationQuestion{"invalid": {Type: "choice", Instructions: "judge", Criteria: json.RawMessage(`{}`)}}
			if _, err := model.Evaluate(context.Background(), bad); err == nil || calls != 1 {
				t.Fatal("invalid input was dispatched")
			}
			cancelled, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := model.Evaluate(cancelled, input); !errors.Is(err, context.Canceled) || calls != 1 {
				t.Fatal("cancellation was not preserved")
			}
		})
	}
}
