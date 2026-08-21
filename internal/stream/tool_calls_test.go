package stream

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolCallTracker(t *testing.T) {
	var tracker ToolCallTracker
	accepted, err := tracker.Accept("call-1", "lookup", json.RawMessage(`{"x":1,"y":2}`))
	if err != nil || !accepted {
		t.Fatalf("first call accepted=%v error=%v", accepted, err)
	}
	accepted, err = tracker.Accept("call-1", "lookup", json.RawMessage(`{"y":2,"x":1}`))
	if err != nil || accepted {
		t.Fatalf("semantic duplicate accepted=%v error=%v", accepted, err)
	}
	if _, err = tracker.Accept("call-1", "lookup", json.RawMessage(`{"x":2,"y":2}`)); err == nil {
		t.Fatal("conflicting identity was accepted")
	}
	var numeric ToolCallTracker
	if accepted, err := numeric.Accept("call-number", "lookup", json.RawMessage(`{"value":9007199254740992}`)); err != nil || !accepted {
		t.Fatalf("first large integer accepted=%v error=%v", accepted, err)
	}
	if _, err := numeric.Accept("call-number", "lookup", json.RawMessage(`{"value":9007199254740993}`)); err == nil {
		t.Fatal("distinct large integer payloads collided")
	}
}

func TestToolCallTrackerPreservesIdlessCalls(t *testing.T) {
	var tracker ToolCallTracker
	for range 2 {
		accepted, err := tracker.Accept("", "lookup", json.RawMessage(`{"x":1}`))
		if err != nil || !accepted {
			t.Fatalf("id-less call accepted=%v error=%v", accepted, err)
		}
	}
}

func TestToolCallBudgetsFailClosed(t *testing.T) {
	budget := ToolCallBudget{calls: MaxTrackedToolCalls}
	if err := budget.Begin(); err == nil {
		t.Fatal("builder count limit was not enforced")
	}
	budget = ToolCallBudget{argumentBytes: MaxTrackedToolCallBytes - 1}
	var builder strings.Builder
	if err := budget.AppendArguments(&builder, "xx"); err == nil {
		t.Fatal("aggregate argument byte limit was not enforced")
	}
	budget = ToolCallBudget{aliases: MaxTrackedToolAliases}
	if err := budget.AddAlias("alias"); err == nil {
		t.Fatal("alias count limit was not enforced")
	}
	budget = ToolCallBudget{metadataBytes: MaxTrackedToolCallBytes - 1}
	if err := budget.AddMetadata("xx"); err == nil {
		t.Fatal("metadata byte limit was not enforced")
	}
	tracker := ToolCallTracker{retainedBytes: MaxTrackedToolCallBytes - 1}
	if _, err := tracker.Accept("id", "lookup", json.RawMessage(`{}`)); err == nil {
		t.Fatal("identity byte limit was not enforced")
	}
	tracker = ToolCallTracker{observations: MaxTrackedToolCalls}
	if _, err := tracker.Accept("", "lookup", json.RawMessage(`{}`)); err == nil {
		t.Fatal("id-less observation count limit was not enforced")
	}
	tracker = ToolCallTracker{observedBytes: MaxTrackedToolCallBytes - 1}
	if _, err := tracker.Accept("", "lookup", json.RawMessage(`{"x":1}`)); err == nil {
		t.Fatal("id-less observation byte limit was not enforced")
	}
}
