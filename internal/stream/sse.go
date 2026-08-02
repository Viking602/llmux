package stream

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

const DefaultMaxFrameBytes = 4 << 20

type SSEEvent struct {
	Name    string
	Data    string
	ID      string
	Retry   string
	Comment string
}

type SSEReader struct {
	reader   *bufio.Reader
	maxBytes int
}

func NewSSEReader(input io.Reader, maxBytes int) *SSEReader {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxFrameBytes
	}
	return &SSEReader{reader: bufio.NewReader(input), maxBytes: maxBytes}
}

func (reader *SSEReader) Next() (SSEEvent, error) {
	var event SSEEvent
	var data []string
	size := 0
	for {
		line, err := reader.readLine()
		if err != nil {
			if errors.Is(err, io.EOF) {
				if size > 0 {
					event.Data = strings.Join(data, "\n")
					return event, io.ErrUnexpectedEOF
				}
				return SSEEvent{}, io.EOF
			}
			return SSEEvent{}, err
		}
		size += len(line) + 1
		if size > reader.maxBytes {
			return SSEEvent{}, fmt.Errorf("llmux: SSE frame exceeds %d bytes", reader.maxBytes)
		}
		if line == "" {
			if size == 1 {
				continue
			}
			event.Data = strings.Join(data, "\n")
			return event, nil
		}
		if strings.HasPrefix(line, ":") {
			event.Comment = strings.TrimPrefix(line, ":")
			if strings.HasPrefix(event.Comment, " ") {
				event.Comment = event.Comment[1:]
			}
			continue
		}
		field, value := splitField(line)
		switch field {
		case "event":
			event.Name = value
		case "data":
			data = append(data, value)
		case "id":
			event.ID = value
		case "retry":
			event.Retry = value
		}
	}
}

func (reader *SSEReader) readLine() (string, error) {
	line, err := reader.reader.ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}
	if errors.Is(err, io.EOF) && line == "" {
		return "", io.EOF
	}
	line = strings.TrimSuffix(line, "\n")
	line = strings.TrimSuffix(line, "\r")
	// A final unterminated line still belongs to the current frame. Returning
	// it first lets Next classify the unfinished frame on the following EOF.
	return line, nil
}

func splitField(line string) (string, string) {
	field, value, found := strings.Cut(line, ":")
	if !found {
		return line, ""
	}
	if strings.HasPrefix(value, " ") {
		value = value[1:]
	}
	return field, value
}
