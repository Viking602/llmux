package stream

import "fmt"

const (
	MaxRetainedStateItems = 4096
	MaxRetainedStateBytes = 16 << 20
)

// StateBudget bounds adapter-owned builders and replay state that have not yet
// crossed the public Part stream.
type StateBudget struct {
	items int
	bytes int
}

func (budget *StateBudget) Retain(bytes int) error {
	if bytes < 0 || budget.items >= MaxRetainedStateItems {
		return fmt.Errorf("retained stream state item limit exceeded (%d)", MaxRetainedStateItems)
	}
	if bytes > MaxRetainedStateBytes-budget.bytes {
		return fmt.Errorf("retained stream state storage limit exceeded (%d bytes)", MaxRetainedStateBytes)
	}
	budget.items++
	budget.bytes += bytes
	return nil
}
