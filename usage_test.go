package llmux

import (
	"math"
	"testing"
)

func TestNormalizeUsagePreservesExplicitZeroAndInclusiveTotals(t *testing.T) {
	usage := NormalizeUsage(Usage{
		InputTokens:                   8,
		CachedInputTokensReported:     true,
		CacheWriteInputTokensReported: true,
		OutputTokens:                  3,
	})
	if usage.TotalTokens != 11 || !usage.CachedInputTokensReported || !usage.CacheWriteInputTokensReported {
		t.Fatalf("normalized usage = %#v", usage)
	}
}

func TestAddUsageSaturatesAndKeepsReportedCacheCounters(t *testing.T) {
	usage := AddUsage(
		Usage{InputTokens: math.MaxInt, CachedInputTokensReported: true},
		Usage{InputTokens: 10, OutputTokens: 5},
	)
	if usage.InputTokens != math.MaxInt || usage.TotalTokens != math.MaxInt || !usage.CachedInputTokensReported {
		t.Fatalf("saturated usage = %#v", usage)
	}
}
