package stream

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateJSONComplexityRejectsTokenAndDepthExpansion(t *testing.T) {
	many := "[" + strings.Repeat("0,", MaxJSONTokens) + "0]"
	if err := ValidateJSONComplexity([]byte(many)); !errors.Is(err, ErrJSONComplexity) {
		t.Fatalf("token limit error = %v", err)
	}
	deep := strings.Repeat("[", MaxJSONDepth+1) + "0" + strings.Repeat("]", MaxJSONDepth+1)
	if err := ValidateJSONComplexity([]byte(deep)); !errors.Is(err, ErrJSONComplexity) {
		t.Fatalf("depth limit error = %v", err)
	}
}
