package llmux

import (
	"fmt"
	"maps"
	"strings"
)

// ConformResult applies the same usage, tool identity, and ownership invariants
// to non-streaming provider responses that ConformStream applies incrementally.
func ConformResult(result Result) (Result, error) {
	result.Usage = NormalizeUsage(result.Usage)
	result.ProviderState = append([]byte(nil), result.ProviderState...)
	result.Raw = append([]byte(nil), result.Raw...)
	result.Warnings = append([]string(nil), result.Warnings...)
	result.Response.Headers = maps.Clone(result.Response.Headers)

	seen := make(map[string]struct{}, len(result.ToolCalls))
	for index := range result.ToolCalls {
		result.ToolCalls[index].Arguments = append([]byte(nil), result.ToolCalls[index].Arguments...)
		if strings.TrimSpace(result.ToolCalls[index].ID) == "" {
			for {
				generated, err := NewGeneratedToolCallID("llmux")
				if err != nil {
					return Result{}, fmt.Errorf("generate tool call id: %w", err)
				}
				if _, duplicate := seen[generated]; !duplicate {
					result.ToolCalls[index].ID = generated
					break
				}
			}
		}
		if _, duplicate := seen[result.ToolCalls[index].ID]; duplicate {
			return Result{}, fmt.Errorf("duplicate tool call id %q", result.ToolCalls[index].ID)
		}
		seen[result.ToolCalls[index].ID] = struct{}{}
	}

	result.Content = cloneResultContent(result.Content)
	toolIndex := 0
	for index := range result.Content {
		if result.Content[index].Kind != ContentToolCall || toolIndex >= len(result.ToolCalls) {
			continue
		}
		call := result.ToolCalls[toolIndex]
		result.Content[index].ToolCall = &call
		toolIndex++
	}
	if result.FinishReason == "" {
		result.FinishReason = FinishUnknown
	}
	return result, nil
}

func cloneResultContent(content []ContentPart) []ContentPart {
	cloned := make([]ContentPart, len(content))
	for index, part := range content {
		cloned[index] = part
		cloned[index].Data = append([]byte(nil), part.Data...)
		cloned[index].ProviderData = append([]byte(nil), part.ProviderData...)
		if part.ToolCall != nil {
			call := *part.ToolCall
			call.Arguments = append([]byte(nil), call.Arguments...)
			cloned[index].ToolCall = &call
		}
		if part.ToolResult != nil {
			result := *part.ToolResult
			result.Structured = append([]byte(nil), result.Structured...)
			cloned[index].ToolResult = &result
		}
		if part.Source != nil {
			source := *part.Source
			cloned[index].Source = &source
		}
	}
	return cloned
}
