package stream

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const (
	MaxJSONTokens = 65_536
	MaxJSONDepth  = 128
)

var ErrJSONComplexity = errors.New("stream JSON exceeds safe structural limits")

// ValidateJSONComplexity bounds decoded descriptor expansion before a provider
// unmarshals an SSE frame into slices and maps.
func ValidateJSONComplexity(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	depth := 0
	for tokens := 1; ; tokens++ {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if tokens > MaxJSONTokens {
			return fmt.Errorf("%w: more than %d tokens", ErrJSONComplexity, MaxJSONTokens)
		}
		delimiter, ok := token.(json.Delim)
		if !ok {
			continue
		}
		switch delimiter {
		case '{', '[':
			depth++
			if depth > MaxJSONDepth {
				return fmt.Errorf("%w: depth exceeds %d", ErrJSONComplexity, MaxJSONDepth)
			}
		case '}', ']':
			depth--
		}
	}
}
