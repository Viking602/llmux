package typesafe

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/provider/catalog"
)

func sampleRequest() llmux.EvaluationRequest {
	return llmux.EvaluationRequest{
		State: map[string]any{"message": "I was charged twice.", "attempts": 2, "resolved": false},
		Questions: map[string]llmux.EvaluationQuestion{
			"team":     {Type: llmux.QuestionChoice, Instructions: map[string]string{"task": "Choose a team"}, Criteria: map[string]any{"billing": nil, "support": "Technical problems"}},
			"severity": {Type: llmux.QuestionScore, Instructions: []string{"Rate urgency"}, Criteria: []any{"Can wait", map[string]string{"level": "Needs attention"}}},
			"spam":     {Type: llmux.QuestionNoul, Instructions: "Is this spam?"},
		},
	}
}

const sampleResponse = `{"model":"jev-1.13.0","answers":{
	"team":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.9,"support":0.1}},
	"severity":{"type":"score","score":0.25,"confidence":0.5,"probabilities":{"0":0.75,"1":0.25},"legend":{"0":"Can wait","1":{"level":"Needs attention"}}},
	"spam":{"type":"noul","noul":0}
},"usage":{"input_tokens":120,"output_tokens":12}}`

func TestSystemOneRoundTrip(t *testing.T) {
	var paths []string
	headers := http.Header{"X-Client": {"original"}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer test-key" || r.Header.Get("X-Client") != "original" {
			t.Error("missing authentication or config headers were not cloned")
		}
		switch r.URL.Path {
		case "/v1/systemone":
			if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" || r.Header.Get("X-Request") != "per-call" {
				t.Error("incorrect evaluation method or headers")
			}
			var body map[string]json.RawMessage
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if len(body) != 3 || string(body["model"]) != `"jev-latest"` {
				t.Errorf("wrong request shape: %s", body)
			}
			want, _ := json.Marshal(sampleRequest().Questions)
			if string(body["questions"]) != string(want) {
				t.Errorf("questions changed: %s", body["questions"])
			}
			want, _ = json.Marshal(sampleRequest().State)
			if string(body["state"]) != string(want) {
				t.Errorf("state changed: %s", body["state"])
			}
			_, _ = io.WriteString(w, sampleResponse)
		case "/v1/models":
			if r.Method != http.MethodGet {
				t.Error("models must use GET")
			}
			_, _ = io.WriteString(w, `{"models":[{"name":"jev-latest","description":"System One","release_date":"2026-09-15"},{"name":"jev-preview","description":"Preview","release_date":"2026-09-15"}]}`)
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL + "/v1/", Headers: headers, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	headers.Set("X-Client", "mutated")
	model, err := llmux.OpenEvaluationModel(provider, DefaultModel)
	if err != nil || model.ModelID() != DefaultModel {
		t.Fatalf("model: %v, %v", model, err)
	}
	request := sampleRequest()
	request.Headers = map[string]string{"X-Request": "per-call", "Authorization": "must-not-replace-config-key"}
	result, err := model.Evaluate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Response.ModelID != "jev-1.13.0" || result.Usage.InputTokens != 120 || result.Usage.OutputTokens != 12 || result.Usage.TotalTokens != 132 || string(result.Raw) != sampleResponse {
		t.Fatalf("lost response metadata: %#v", result)
	}
	if result.Answers["spam"].Noul == nil || *result.Answers["spam"].Noul != 0 || *result.Answers["team"].Choice != "billing" || result.Answers["team"].Probabilities["support"] != 0.1 || *result.Answers["severity"].Score != 0.25 {
		t.Fatalf("lost answers: %#v", result.Answers)
	}
	if result.Answers["severity"].Legend["1"].(map[string]any)["level"] != "Needs attention" {
		t.Fatal("lost structured legend")
	}
	models, err := llmux.ListModels(context.Background(), provider)
	if err != nil || len(models) != 2 || models[0].ID != DefaultModel || models[0].Created != time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC).Unix() {
		t.Fatalf("models = %#v, %v", models, err)
	}
	caps := models[0].Capabilities
	if caps.Streaming == nil || *caps.Streaming || caps.ToolCalling == nil || *caps.ToolCalling || !*caps.StructuredOutput || caps.OutputModalities[0] != llmux.ModalityEvaluation {
		t.Fatalf("model capabilities: %#v", caps)
	}
	descriptor, err := llmux.DescribeProvider(provider)
	listed, ok := catalog.Lookup("typesafe")
	if err != nil || !ok || !reflect.DeepEqual(descriptor.Capabilities, listed.Descriptor().Capabilities) || !reflect.DeepEqual(descriptor.WireProtocols, listed.Descriptor().WireProtocols) {
		t.Fatalf("descriptor/catalog drift: %#v, %#v, %v", descriptor, listed, err)
	}
	for _, capability := range descriptor.Capabilities {
		if capability == llmux.CapabilityLanguage {
			t.Fatal("Jev advertised language generation")
		}
	}
	_, err = provider.LanguageModel(DefaultModel)
	assertErrorKind(t, err, llmux.ErrorUnsupported)
	if !reflect.DeepEqual(paths, []string{"/v1/systemone", "/v1/models"}) {
		t.Fatalf("paths = %v", paths)
	}
}

func TestRequestValidation(t *testing.T) {
	tooManyChoices := make(map[string]any)
	for i := 0; i < 256; i++ {
		tooManyChoices[string(rune(i))] = nil
	}
	for name, q := range map[string]llmux.EvaluationQuestion{
		"unknown":                    {Type: "chat"},
		"scalar instructions":        {Type: llmux.QuestionNoul, Instructions: true},
		"empty choice":               {Type: llmux.QuestionChoice},
		"array choice":               {Type: llmux.QuestionChoice, Criteria: []string{"a", "b"}},
		"too many choices":           {Type: llmux.QuestionChoice, Criteria: tooManyChoices},
		"numeric choice description": {Type: llmux.QuestionChoice, Criteria: map[string]any{"a": 3}},
		"empty score":                {Type: llmux.QuestionScore, Criteria: []string{}},
		"too many levels":            {Type: llmux.QuestionScore, Criteria: make([]string, 11)},
		"null score level":           {Type: llmux.QuestionScore, Criteria: []any{nil}},
		"wrong noul keys":            {Type: llmux.QuestionNoul, Criteria: map[string]any{"yes": "yes"}},
		"invalid noul criteria":      {Type: llmux.QuestionNoul, Criteria: []string{"yes"}},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, err := buildRequest(DefaultModel, llmux.EvaluationRequest{State: "state", Questions: map[string]llmux.EvaluationQuestion{"q": q}})
			if err == nil {
				t.Fatal("accepted invalid question")
			}
		})
	}
	for _, state := range []any{nil, true, 3, json.RawMessage(`null`), json.RawMessage(`{`), make(chan int)} {
		r := sampleRequest()
		r.State = state
		if _, _, err := buildRequest(DefaultModel, r); err == nil {
			t.Fatalf("accepted invalid state %T", state)
		}
	}
	for _, state := range []any{"", []string{"text"}, struct{ Text string }{"test"}, json.RawMessage(`{"nested":null}`)} {
		r := sampleRequest()
		r.State = state
		if _, _, err := buildRequest(DefaultModel, r); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := buildRequest(DefaultModel, llmux.EvaluationRequest{State: "test"}); err == nil {
		t.Fatal("accepted empty questions")
	}
	// Match the live OpenAPI schema: nullable instructions and a one-level score.
	_, _, err := buildRequest(DefaultModel, llmux.EvaluationRequest{State: "test", Questions: map[string]llmux.EvaluationQuestion{
		"q": {Type: llmux.QuestionScore, Criteria: []string{"Only level"}},
		"n": {Type: llmux.QuestionNoul, Criteria: map[string]any{"true": map[string]any{"tags": []string{"urgent"}}, "false": nil}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, config := range []Config{{}, {APIKey: " "}, {APIKey: "key\n"}, {APIKey: "key", BaseURL: "ftp://example.com"}, {APIKey: "key", BaseURL: "https://user:secret@example.com"}, {APIKey: "key", BaseURL: "https://example.com?key=secret"}} {
		if _, err := New(config); err == nil {
			t.Fatal("accepted invalid config")
		}
	}
	provider, err := New(Config{APIKey: "key"})
	if err != nil || provider.config.BaseURL != DefaultBaseURL {
		t.Fatalf("defaults: %v", err)
	}
	if _, err := provider.EvaluationModel(" "); err == nil {
		t.Fatal("accepted empty model")
	}
	if _, err := provider.EvaluationModel("jev-future-version"); err != nil {
		t.Fatal(err)
	}
	if _, err := llmux.OpenEvaluationModel(nil, DefaultModel); err == nil {
		t.Fatal("accepted nil provider")
	}
}

func TestRejectMalformedResponses(t *testing.T) {
	_, questions, err := buildRequest(DefaultModel, sampleRequest())
	if err != nil {
		t.Fatal(err)
	}
	for _, replacement := range [][2]string{
		{`"jev-1.13.0"`, `""`}, {`"usage":{`, `"missing_usage":{`},
		{`"input_tokens":120`, `"input_tokens":null`}, {`"output_tokens":12`, `"output_tokens":-1`},
		{`"spam":`, `"unexpected":`}, {`"type":"noul"`, `"type":"choice"`},
		{`"noul":0`, `"noul":null`}, {`"noul":0`, `"noul":2`},
		{`"confidence":0.8`, `"confidence":null`}, {`"choice":"billing"`, `"choice":"other"`},
		{`"billing":0.9`, `"billing":null`}, {`"billing":0.9`, `"billing":0.2`},
		{`"support":0.1`, `"unexpected":0.1`}, {`"score":0.25`, `"score":null`},
		{`"score":0.25`, `"score":2`}, {`"1":{"level":"Needs attention"}`, `"2":"wrong index"`},
	} {
		t.Run(replacement[0]+"->"+replacement[1], func(t *testing.T) {
			payload := strings.Replace(sampleResponse, replacement[0], replacement[1], 1)
			if _, err := parseResult([]byte(payload), questions); err == nil {
				t.Fatal("accepted malformed response")
			}
		})
	}
	for _, payload := range []string{`{}`, `null`, sampleResponse + `{}`, sampleResponse[:20]} {
		if _, err := parseResult([]byte(payload), questions); err == nil {
			t.Fatal("accepted malformed envelope")
		}
	}
}

func TestHTTPFailuresAndRetries(t *testing.T) {
	for _, test := range []struct {
		status   int
		body     string
		kind     llmux.ErrorKind
		attempts int
	}{
		{401, `{"detail":"Invalid key"}`, llmux.ErrorAuthentication, 1},
		{403, `{"detail":{"error_type":"authentication_error","message":"Must supply an API key!"}}`, llmux.ErrorAuthentication, 1},
		{403, `{"detail":"Not permitted"}`, llmux.ErrorPermission, 1},
		{422, `{"detail":[{"loc":["body","state"],"msg":"Field required"}]}`, llmux.ErrorInvalidRequest, 1},
		{429, `{"detail":"Rate limit"}`, llmux.ErrorRateLimit, 3},
		{529, `{"detail":"Overloaded"}`, llmux.ErrorServer, 3},
	} {
		t.Run(string(test.kind)+"/"+test.body, func(t *testing.T) {
			attempts := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts++
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			defer server.Close()
			provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client(), Retry: llmux.RetryPolicy{BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}})
			if err != nil {
				t.Fatal(err)
			}
			model, _ := provider.EvaluationModel(DefaultModel)
			_, err = model.Evaluate(context.Background(), sampleRequest())
			got := assertErrorKind(t, err, test.kind)
			if attempts != test.attempts || got.StatusCode != test.status || string(got.Raw) != test.body || got.RetryAfter != time.Second {
				t.Fatalf("error=%#v, attempts=%d", got, attempts)
			}
		})
	}
	response := sampleResponse
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(529)
			return
		}
		_, _ = io.WriteString(w, response)
	}))
	defer server.Close()
	provider, _ := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client(), Retry: llmux.RetryPolicy{BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}})
	model, _ := provider.EvaluationModel(DefaultModel)
	if _, err := model.Evaluate(context.Background(), sampleRequest()); err != nil || attempts != 2 {
		t.Fatalf("retry recovery: %v (%d)", err, attempts)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := model.Evaluate(ctx, sampleRequest())
	assertErrorKind(t, err, llmux.ErrorCancelled)
	if !errors.Is(err, context.Canceled) {
		t.Fatal("lost cancellation cause")
	}
	response = `{}` + strings.Repeat(" ", maxResponseBytes)
	_, err = model.Evaluate(context.Background(), sampleRequest())
	assertErrorKind(t, err, llmux.ErrorUnknown)
	for _, body := range []string{`{}`, `null`, `{"models":[{}]}`} {
		response = body
		if _, err := provider.ListModels(context.Background()); err == nil {
			t.Fatal("accepted malformed models")
		}
	}
}

func assertErrorKind(t *testing.T, err error, kind llmux.ErrorKind) *llmux.ProviderError {
	t.Helper()
	var providerErr *llmux.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Kind != kind {
		t.Fatalf("error = %v, want %s", err, kind)
	}
	return providerErr
}

func TestLiveJev(t *testing.T) {
	if os.Getenv("LLMUX_TYPESAFE_LIVE") != "1" {
		t.Skip("set LLMUX_TYPESAFE_LIVE=1 and TYPESAFE_API_KEY for a live call")
	}
	provider, err := New(Config{APIKey: os.Getenv(APIKeyEnvVar)})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	models, err := llmux.ListModels(ctx, provider)
	if err != nil || len(models) == 0 {
		t.Fatalf("live models: %v", err)
	}
	model, err := llmux.OpenEvaluationModel(provider, DefaultModel)
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Evaluate(ctx, sampleRequest())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model=%s, answers=%d, input_tokens=%d, output_tokens=%d", result.Response.ModelID, len(result.Answers), result.Usage.InputTokens, result.Usage.OutputTokens)
}
