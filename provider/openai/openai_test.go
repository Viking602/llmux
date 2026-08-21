package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Viking602/llmux"
)

func TestChatGenerateAndStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["model"] != "gpt-test" {
			t.Errorf("model = %#v", body["model"])
		}
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(response, "data: {\"id\":\"chat-1\",\"model\":\"gpt-test\",\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n")
			_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = fmt.Fprint(response, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n")
			_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
			return
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(response, `{"id":"chat-1","model":"gpt-test","created":1,"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client(), WireAPI: ChatCompletions})
	if err != nil {
		t.Fatal(err)
	}
	model, err := provider.LanguageModel("gpt-test")
	if err != nil {
		t.Fatal(err)
	}
	request := llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hi")}}
	result, err := model.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.FinishReason != llmux.FinishStop || result.Usage.TotalTokens != 3 {
		t.Fatalf("result = %#v", result)
	}
	stream, err := model.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	result, err = llmux.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.FinishReason != llmux.FinishStop || result.Usage.TotalTokens != 3 {
		t.Fatalf("stream result = %#v", result)
	}
}

func TestResponsesPreservesProviderStateAndToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses" {
			t.Errorf("path = %q", request.URL.Path)
		}
		_, _ = fmt.Fprint(response, `{"id":"resp-1","model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg-1","content":[{"type":"output_text","text":"done"}]},{"type":"function_call","id":"item-1","call_id":"call-1","name":"lookup","arguments":"{\"id\":7}"}],"usage":{"prompt_tokens":4,"completion_tokens":3}}`)
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	result, err := model.Generate(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "done" || len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "call-1" || !json.Valid(result.ProviderState) || result.Usage.TotalTokens != 7 {
		t.Fatalf("result = %#v", result)
	}
}

func TestProtectedBodyFieldsCannotBeOverridden(t *testing.T) {
	provider, err := New(Config{APIKey: "test-key"})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	_, err = model.Generate(context.Background(), llmux.Request{Options: llmux.CallOptions{BodyOverrides: json.RawMessage(`{"model":"stolen"}`)}})
	if err == nil {
		t.Fatal("expected protected body override error")
	}
}

func TestResponsesStreamPreservesProviderFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"error\",\"code\":\"server_is_overloaded\",\"message\":\"retry me\"}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", Endpoint: server.URL, Client: server.Client(), WireAPI: Responses})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	part, err := stream.Recv()
	if err != nil || part.Kind != llmux.PartError || part.Err == nil || !strings.Contains(part.Err.Error(), "retry me") {
		t.Fatalf("part=%#v error=%v", part, err)
	}
}

func TestResponsesStreamAcceptsEOFAfterCompleted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\",\"output\":[],\"usage\":{}}}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", Endpoint: server.URL, Client: server.Client(), WireAPI: Responses})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil || result.Text != "done" || result.FinishReason != llmux.FinishStop {
		t.Fatalf("result=%#v error=%v", result, err)
	}
}

func TestResponsesUsagePreservesReportedZero(t *testing.T) {
	result, err := parseResponsesResult([]byte(`{"status":"completed","output":[],"usage":{"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Usage.CachedInputTokensReported || !result.Usage.CacheWriteInputTokensReported {
		t.Fatalf("usage=%#v", result.Usage)
	}
}

func TestResponsesStreamCorrelatesProvisionalAndCanonicalToolIDs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_tmp_read\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"item_id\":\"fc_tmp_read\",\"arguments\":\"{\\\"path\\\":\\\"docs/desktop.md\\\"}\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_tmp_read\",\"call_id\":\"call-final\",\"name\":\"lookup\",\"arguments\":\"{\\\"path\\\":\\\"docs/desktop.md\\\"}\"}}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_tmp_read\",\"call_id\":\"call-final\",\"name\":\"lookup\",\"arguments\":\"{\\\"path\\\":\\\"docs/desktop.md\\\"}\"}],\"usage\":{}}}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", Endpoint: server.URL, Client: server.Client(), WireAPI: Responses})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "go")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "call-final" {
		t.Fatalf("tool calls = %#v", result.ToolCalls)
	}
}

func TestChatStreamToolFinalizationIsIdempotent(t *testing.T) {
	builder := &toolBuilder{id: "call-1", name: "lookup"}
	builder.arguments.WriteString(`{"x":1}`)
	stream := &chatStream{tools: map[int]*toolBuilder{7: builder}, finish: llmux.FinishToolCalls}
	stream.finishParts()
	stream.finishParts()
	count := 0
	for _, part := range stream.pending {
		if part.Kind == llmux.PartToolCall {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("tool call parts = %d, want 1", count)
	}
}

func TestChatStreamDeduplicatesStableCallIDAcrossIndexes(t *testing.T) {
	first := &toolBuilder{id: "call-1", name: "lookup"}
	first.arguments.WriteString(`{"x":1}`)
	second := &toolBuilder{id: "call-1", name: "lookup"}
	second.arguments.WriteString(`{"x":1}`)
	stream := &chatStream{tools: map[int]*toolBuilder{0: first, 1: second}, finish: llmux.FinishToolCalls}
	stream.finishParts()
	counts := make(map[llmux.PartKind]int)
	for _, part := range stream.pending {
		counts[part.Kind]++
	}
	if counts[llmux.PartToolCall] != 1 || counts[llmux.PartError] != 0 {
		t.Fatalf("chat parts = %#v", counts)
	}

	conflicting := &toolBuilder{id: "call-1", name: "lookup"}
	conflicting.arguments.WriteString(`{"x":2}`)
	conflictStream := &chatStream{tools: map[int]*toolBuilder{0: first, 1: conflicting}, finish: llmux.FinishToolCalls}
	conflictStream.finishParts()
	errorCount := 0
	for _, part := range conflictStream.pending {
		if part.Kind == llmux.PartError {
			errorCount++
		}
	}
	if errorCount != 1 {
		t.Fatalf("conflicting chat tool identity errors = %d, want 1", errorCount)
	}
}

func newTestResponsesStream() *responsesStream {
	return &responsesStream{
		toolsByOutputIndex:    make(map[int]*toolBuilder),
		toolsByItemID:         make(map[string]*toolBuilder),
		toolsByCallID:         make(map[string]*toolBuilder),
		toolsByPrefixedCallID: make(map[string]*toolBuilder),
	}
}

func mustResponsesTool(t *testing.T, stream *responsesStream, itemID, callID string, index *int) *toolBuilder {
	t.Helper()
	builder, err := stream.tool(itemID, callID, index)
	if err != nil {
		t.Fatal(err)
	}
	return builder
}

func TestResponsesToolIdentityAliasesAndParallelIndexes(t *testing.T) {
	stream := newTestResponsesStream()
	provisional := mustResponsesTool(t, stream, "fc_tmp_read", "", nil)
	if got := mustResponsesTool(t, stream, "fc_tmp_read", "call-final", nil); got != provisional {
		t.Fatal("final call ID created a second builder")
	}
	if got := mustResponsesTool(t, stream, "fc_call-final", "", nil); got != provisional {
		t.Fatal("prefixed call ID alias did not resolve the original builder")
	}
	if got := mustResponsesTool(t, stream, "", "call-final", nil); got != provisional {
		t.Fatal("canonical call ID did not resolve the original builder")
	}

	firstIndex, secondIndex := 0, 1
	first := mustResponsesTool(t, stream, "item-1", "call-1", &firstIndex)
	second := mustResponsesTool(t, stream, "item-2", "call-2", &secondIndex)
	if first == second ||
		mustResponsesTool(t, stream, "", "", &firstIndex) != first ||
		mustResponsesTool(t, stream, "", "", &secondIndex) != second {
		t.Fatal("parallel output indexes were not isolated")
	}

	abandoned := mustResponsesTool(t, stream, "item-abandoned", "", nil)
	if err := stream.emitPendingTools(); err != nil {
		t.Fatal(err)
	}
	if abandoned.emitted {
		t.Fatal("terminal fallback emitted an empty unfinished tool call")
	}
	fallback := mustResponsesTool(t, stream, "item-fallback", "call-fallback", nil)
	fallback.name = "lookup"
	fallback.arguments.WriteString(`{"x":1}`)
	fallback.argumentsDone = true
	if err := stream.emitPendingTools(); err != nil {
		t.Fatal(err)
	}
	if !fallback.emitted {
		t.Fatal("completed arguments were not emitted by terminal fallback")
	}
	fallback.arguments.Reset()
	fallback.arguments.WriteString(`{"x":2}`)
	if err := stream.emitTool(fallback); err == nil {
		t.Fatal("conflicting final payload reused one Responses identity")
	}
	fallback.arguments.Reset()
	fallback.arguments.WriteString(`{"x":1}`)
	fallback.id = "call-changed"
	if err := stream.emitTool(fallback); err == nil {
		t.Fatal("emitted Responses identity changed its final call ID")
	}
}

func TestResponsesToolAliasConflictsFailClosed(t *testing.T) {
	stream := newTestResponsesStream()
	firstIndex, secondIndex := 0, 1
	mustResponsesTool(t, stream, "item-1", "call-1", &firstIndex)
	mustResponsesTool(t, stream, "item-2", "call-2", &secondIndex)
	if _, err := stream.tool("item-2", "call-1", &secondIndex); err == nil {
		t.Fatal("conflicting output-index and call-ID aliases were merged")
	}
	if _, err := stream.tool("item-1", "call-changed", &firstIndex); err == nil {
		t.Fatal("one Responses identity changed call IDs")
	}
}

func TestResponsesToolPrefixedAliasCanArriveBeforeCallID(t *testing.T) {
	stream := newTestResponsesStream()
	provisional := mustResponsesTool(t, stream, "fc_call-final", "", nil)
	if got := mustResponsesTool(t, stream, "", "call-final", nil); got != provisional {
		t.Fatal("prefixed item alias created a second builder before canonical call ID")
	}
}

func TestResponsesToolRawAliasCanArriveBeforeCallID(t *testing.T) {
	stream := newTestResponsesStream()
	rawAlias := mustResponsesTool(t, stream, "call-raw", "", nil)
	if got := mustResponsesTool(t, stream, "", "call-raw", nil); got != rawAlias {
		t.Fatal("raw item/call ID alias created a second builder")
	}
}

func TestResponsesCustomToolInputEmitsOnce(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_tmp\",\"name\":\"apply_patch\",\"input\":\"\"}}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"item_id\":\"ctc_tmp\",\"input\":\"PATCH\"}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_tmp\",\"call_id\":\"call-custom\",\"name\":\"apply_patch\",\"input\":\"PATCH\"}}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\",\"output\":[{\"type\":\"custom_tool_call\",\"id\":\"ctc_tmp\",\"call_id\":\"call-custom\",\"name\":\"apply_patch\",\"input\":\"PATCH\"}],\"usage\":{}}}\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", Endpoint: server.URL, Client: server.Client(), WireAPI: Responses})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := llmux.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "call-custom" || string(result.ToolCalls[0].Arguments) != `{"input":"PATCH"}` {
		t.Fatalf("custom tool calls = %#v", result.ToolCalls)
	}

	result, err = parseResponsesResult([]byte(`{"status":"completed","output":[{"type":"custom_tool_call","id":"ctc_tmp","call_id":"call-custom","name":"apply_patch","input":"PATCH"}]}`))
	if err != nil || len(result.ToolCalls) != 1 || string(result.ToolCalls[0].Arguments) != `{"input":"PATCH"}` {
		t.Fatalf("non-stream custom tool result=%#v error=%v", result, err)
	}
}

func TestResponsesCustomToolDeltaFallbackWrapsInput(t *testing.T) {
	stream := newTestResponsesStream()
	index := 0
	added := json.RawMessage(`{"type":"custom_tool_call","id":"ctc_tmp","name":"apply_patch"}`)
	if err := stream.mapEvent("response.output_item.added", "", "", "", "", "", &index, nil, added, nil, added); err != nil {
		t.Fatal(err)
	}
	if err := stream.mapEvent("response.custom_tool_call_input.delta", "PATCH", "apply_patch", "", "ctc_tmp", "", &index, nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	done := json.RawMessage(`{"type":"custom_tool_call","id":"ctc_tmp","call_id":"call-custom","name":"apply_patch"}`)
	if err := stream.mapEvent("response.output_item.done", "", "", "", "", "", &index, nil, done, nil, done); err != nil {
		t.Fatal(err)
	}
	for _, part := range stream.pending {
		if part.Kind == llmux.PartToolCall {
			if part.ToolCall == nil || string(part.ToolCall.Arguments) != `{"input":"PATCH"}` {
				t.Fatalf("custom fallback tool call = %#v", part.ToolCall)
			}
			return
		}
	}
	t.Fatal("custom fallback omitted tool call")
}

func TestResponsesDefersItemOnlyCallUntilCompleted(t *testing.T) {
	stream := newTestResponsesStream()
	index := 0
	itemDone := json.RawMessage(`{"type":"function_call","id":"item-only","name":"lookup","arguments":"{}"}`)
	if err := stream.mapEvent("response.output_item.done", "", "", "", "", "", &index, nil, itemDone, nil, itemDone); err != nil {
		t.Fatal(err)
	}
	for _, part := range stream.pending {
		if part.Kind == llmux.PartToolCall {
			t.Fatal("item-only call executed before canonical terminal correlation")
		}
	}
	completed := json.RawMessage(`{"status":"completed","output":[{"type":"function_call","id":"item-only","call_id":"call-final","name":"lookup","arguments":"{}"}]}`)
	if err := stream.mapEvent("response.completed", "", "", "", "", "", nil, nil, nil, completed, completed); err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, part := range stream.pending {
		if part.Kind == llmux.PartToolCall {
			count++
			if part.ToolCall == nil || part.ToolCall.ID != "call-final" {
				t.Fatalf("terminal tool call = %#v", part.ToolCall)
			}
		}
	}
	if count != 1 {
		t.Fatalf("terminal tool calls = %d, want 1", count)
	}
}

func TestResponsesRejectsConflictingAndMalformedTerminalCalls(t *testing.T) {
	stream := newTestResponsesStream()
	builder := mustResponsesTool(t, stream, "item-1", "call-1", nil)
	builder.name = "lookup"
	if err := stream.finalizeToolArguments(builder, "", json.RawMessage(`{"x":1}`)); err != nil {
		t.Fatal(err)
	}
	if err := stream.finalizeToolArguments(builder, "", json.RawMessage(`{"x":2}`)); err == nil {
		t.Fatal("conflicting arguments.done payload was accepted")
	}
	if err := stream.finalizeToolArguments(builder, "delete", json.RawMessage(`{"x":1}`)); err == nil {
		t.Fatal("conflicting terminal tool name was accepted")
	}

	for _, response := range []json.RawMessage{
		json.RawMessage(`{"status":"completed","output":[{"type":"function_call","id":"item-2","call_id":"call-2","name":"lookup","arguments":"not-json"}]}`),
		json.RawMessage(`{"status":"completed","output":[{"type":"function_call","id":123,"call_id":"call-2","name":"lookup","arguments":"{}"}]}`),
	} {
		candidate := newTestResponsesStream()
		if err := candidate.mapEvent("response.completed", "", "", "", "", "", nil, nil, nil, response, response); err == nil {
			t.Fatalf("malformed completed response was accepted: %s", response)
		}
	}
}

func TestResponsesRejectsToolMutationAfterCompletion(t *testing.T) {
	stream := newTestResponsesStream()
	first := json.RawMessage(`{"status":"completed","output":[{"type":"function_call","id":"item-1","call_id":"call-1","name":"lookup","arguments":"{}"}]}`)
	if err := stream.mapEvent("response.completed", "", "", "", "", "", nil, nil, nil, first, first); err != nil {
		t.Fatal(err)
	}
	changed := json.RawMessage(`{"status":"completed","output":[{"type":"function_call","id":"item-1","call_id":"call-1","name":"lookup","arguments":"{}"},{"type":"function_call","id":"item-2","call_id":"call-2","name":"lookup","arguments":"{}"}]}`)
	if err := stream.mapEvent("response.completed", "", "", "", "", "", nil, nil, nil, changed, changed); err == nil {
		t.Fatal("repeated response.completed introduced a new tool call")
	}
	item := json.RawMessage(`{"type":"function_call","id":"item-2","call_id":"call-2","name":"lookup","arguments":"{}"}`)
	if err := stream.mapEvent("response.output_item.done", "", "", "", "", "", nil, nil, item, nil, item); err == nil {
		t.Fatal("tool event after response.completed was accepted")
	}
	arguments := json.RawMessage(`"{\"x\":1}"`)
	if err := stream.mapEvent("response.function_call_arguments.done", "", "lookup", "", "item-3", "call-3", nil, arguments, nil, nil, nil); err == nil {
		t.Fatal("arguments.done introduced a tool call after response.completed")
	}
}

func TestResponsesRejectsUncommittedBuilderAfterCompletion(t *testing.T) {
	stream := newTestResponsesStream()
	index := 0
	added := json.RawMessage(`{"type":"function_call","id":"item-late","name":"lookup"}`)
	if err := stream.mapEvent("response.output_item.added", "", "", "", "", "", &index, nil, added, nil, added); err != nil {
		t.Fatal(err)
	}
	completed := json.RawMessage(`{"status":"completed","output":[]}`)
	if err := stream.mapEvent("response.completed", "", "", "", "", "", nil, nil, nil, completed, completed); err != nil {
		t.Fatal(err)
	}
	done := json.RawMessage(`{"type":"function_call","id":"item-late","call_id":"call-late","name":"lookup","arguments":"{}"}`)
	repeated := json.RawMessage(`{"status":"completed","output":[{"type":"function_call","id":"item-late","call_id":"call-late","name":"lookup","arguments":"{}"}]}`)
	if err := stream.mapEvent("response.completed", "", "", "", "", "", nil, nil, nil, repeated, repeated); err == nil {
		t.Fatal("repeated response.completed finalized an uncommitted builder")
	}
	if err := stream.mapEvent("response.output_item.done", "", "", "", "", "", &index, nil, done, nil, done); err == nil {
		t.Fatal("uncommitted builder emitted after response.completed")
	}
}

func TestChatStreamRejectsToolCallIDDrift(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":\"}}]},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-2\",\"function\":{\"name\":\"\",\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n")
		_, _ = fmt.Fprint(response, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider, err := New(Config{APIKey: "test-key", BaseURL: server.URL, Client: server.Client(), WireAPI: ChatCompletions})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := provider.LanguageModel("gpt-test")
	stream, err := model.Stream(context.Background(), llmux.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := llmux.Collect(stream); err == nil {
		t.Fatal("chat tool index changed call IDs")
	}
}
