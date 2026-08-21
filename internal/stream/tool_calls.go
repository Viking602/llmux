package stream

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const (
	MaxTrackedToolCalls     = 4096
	MaxTrackedToolAliases   = MaxTrackedToolCalls * 4
	MaxTrackedToolCallBytes = 16 << 20
	MaxToolArgumentBytes    = 4 << 20
	MaxToolMetadataBytes    = 1 << 20
)

// ToolCallTracker makes protocol terminal events idempotent by stable wire
// identity. Empty identities are intentionally not deduplicated because some
// protocols allow id-less parallel calls that cannot be safely correlated.
type ToolCallTracker struct {
	signatures    map[string]string
	retainedBytes int
	observations  int
	observedBytes int
}

// Accept records one completed tool call. It reports true for the first
// observation, false for an exact semantic duplicate, and an error when a
// provider reuses the same identity for a different call.
func (tracker *ToolCallTracker) Accept(identity, name string, arguments json.RawMessage) (bool, error) {
	if tracker.observations >= MaxTrackedToolCalls {
		return false, fmt.Errorf("tool call observation limit exceeded (%d)", MaxTrackedToolCalls)
	}
	observed := len(identity) + len(name) + len(arguments)
	if observed > MaxTrackedToolCallBytes-tracker.observedBytes {
		return false, fmt.Errorf("tool call observation storage limit exceeded (%d bytes)", MaxTrackedToolCallBytes)
	}
	tracker.observations++
	tracker.observedBytes += observed
	if identity == "" {
		return true, nil
	}
	signature, err := toolCallSignature(name, arguments)
	if err != nil {
		return false, err
	}
	if previous, exists := tracker.signatures[identity]; exists {
		if previous != signature {
			return false, fmt.Errorf("tool call identity %q was reused with conflicting payload", identity)
		}
		return false, nil
	}
	if len(tracker.signatures) >= MaxTrackedToolCalls {
		return false, fmt.Errorf("tool call identity limit exceeded (%d)", MaxTrackedToolCalls)
	}
	retained := len(identity) + len(signature)
	if retained > MaxTrackedToolCallBytes-tracker.retainedBytes {
		return false, fmt.Errorf("tool call identity storage limit exceeded (%d bytes)", MaxTrackedToolCallBytes)
	}
	if tracker.signatures == nil {
		tracker.signatures = make(map[string]string)
	}
	tracker.signatures[identity] = signature
	tracker.retainedBytes += retained
	return true, nil
}

// Seen reports whether a non-empty stable identity already completed. It lets
// streaming adapters suppress duplicate progressive start/delta events while
// still validating the repeated terminal payload through Accept.
func (tracker *ToolCallTracker) Seen(identity string) bool {
	if identity == "" {
		return false
	}
	_, exists := tracker.signatures[identity]
	return exists
}

// ToolCallBudget bounds provisional builders and their cumulative arguments
// before terminal identity tracking begins.
type ToolCallBudget struct {
	calls         int
	aliases       int
	argumentBytes int
	metadataBytes int
}

func (budget *ToolCallBudget) Begin() error {
	if budget.calls >= MaxTrackedToolCalls {
		return fmt.Errorf("tool call builder limit exceeded (%d)", MaxTrackedToolCalls)
	}
	budget.calls++
	return nil
}

func (budget *ToolCallBudget) AddAlias(alias string) error {
	if alias == "" {
		return nil
	}
	if budget.aliases >= MaxTrackedToolAliases {
		return fmt.Errorf("tool call alias limit exceeded (%d)", MaxTrackedToolAliases)
	}
	if err := budget.AddMetadata(alias); err != nil {
		return err
	}
	budget.aliases++
	return nil
}

func (budget *ToolCallBudget) AddMetadata(value string) error {
	if len(value) > MaxToolMetadataBytes {
		return fmt.Errorf("tool call metadata exceeds %d bytes", MaxToolMetadataBytes)
	}
	if len(value) > MaxTrackedToolCallBytes-budget.metadataBytes {
		return fmt.Errorf("tool call metadata storage limit exceeded (%d bytes)", MaxTrackedToolCallBytes)
	}
	budget.metadataBytes += len(value)
	return nil
}

func (budget *ToolCallBudget) AppendArguments(builder *strings.Builder, delta string) error {
	if builder.Len()+len(delta) > MaxToolArgumentBytes {
		return fmt.Errorf("tool call arguments exceed %d bytes", MaxToolArgumentBytes)
	}
	if len(delta) > MaxTrackedToolCallBytes-budget.argumentBytes {
		return fmt.Errorf("tool call argument storage limit exceeded (%d bytes)", MaxTrackedToolCallBytes)
	}
	builder.WriteString(delta)
	budget.argumentBytes += len(delta)
	return nil
}

func toolCallSignature(name string, arguments json.RawMessage) (string, error) {
	var value any
	decoder := json.NewDecoder(bytes.NewReader(arguments))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return "", fmt.Errorf("tool call %q has invalid JSON arguments: %w", name, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return "", fmt.Errorf("tool call %q has trailing JSON arguments", name)
	}
	canonical, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("canonicalize tool call %q arguments: %w", name, err)
	}
	return name + "\x00" + string(canonical), nil
}
