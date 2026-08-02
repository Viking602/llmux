package bedrock

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

const maxEventBytes = 16 << 20

type eventMessage struct {
	MessageType string
	EventType   string
	Payload     []byte
}

func readEvent(input io.Reader) (eventMessage, error) {
	prelude := make([]byte, 12)
	if _, err := io.ReadFull(input, prelude); err != nil {
		return eventMessage{}, err
	}
	total := int(binary.BigEndian.Uint32(prelude[:4]))
	headersLength := int(binary.BigEndian.Uint32(prelude[4:8]))
	if total < 16 || total > maxEventBytes || headersLength < 0 || 12+headersLength+4 > total {
		return eventMessage{}, errors.New("bedrock: invalid event stream frame length")
	}
	if crc32.ChecksumIEEE(prelude[:8]) != binary.BigEndian.Uint32(prelude[8:12]) {
		return eventMessage{}, errors.New("bedrock: invalid event stream prelude CRC")
	}
	rest := make([]byte, total-12)
	if _, err := io.ReadFull(input, rest); err != nil {
		return eventMessage{}, err
	}
	whole := append(append(make([]byte, 0, total-4), prelude...), rest[:len(rest)-4]...)
	if crc32.ChecksumIEEE(whole) != binary.BigEndian.Uint32(rest[len(rest)-4:]) {
		return eventMessage{}, errors.New("bedrock: invalid event stream message CRC")
	}
	headers := rest[:headersLength]
	message := eventMessage{Payload: append([]byte(nil), rest[headersLength:len(rest)-4]...)}
	for offset := 0; offset < len(headers); {
		nameLength := int(headers[offset])
		offset++
		if offset+nameLength+1 > len(headers) {
			return eventMessage{}, errors.New("bedrock: truncated event header")
		}
		name := string(headers[offset : offset+nameLength])
		offset += nameLength
		typeID := headers[offset]
		offset++
		value, consumed, err := eventHeaderValue(typeID, headers[offset:])
		if err != nil {
			return eventMessage{}, err
		}
		offset += consumed
		switch name {
		case ":message-type":
			message.MessageType = value
		case ":event-type", ":exception-type":
			message.EventType = value
		}
	}
	return message, nil
}

func eventHeaderValue(typeID byte, data []byte) (string, int, error) {
	switch typeID {
	case 0, 1:
		return "", 0, nil
	case 2:
		if len(data) < 1 {
			return "", 0, io.ErrUnexpectedEOF
		}
		return "", 1, nil
	case 3:
		if len(data) < 2 {
			return "", 0, io.ErrUnexpectedEOF
		}
		return "", 2, nil
	case 4:
		if len(data) < 4 {
			return "", 0, io.ErrUnexpectedEOF
		}
		return "", 4, nil
	case 5, 8:
		if len(data) < 8 {
			return "", 0, io.ErrUnexpectedEOF
		}
		return "", 8, nil
	case 6, 7:
		if len(data) < 2 {
			return "", 0, io.ErrUnexpectedEOF
		}
		length := int(binary.BigEndian.Uint16(data[:2]))
		if len(data) < 2+length {
			return "", 0, io.ErrUnexpectedEOF
		}
		if typeID == 7 {
			return string(data[2 : 2+length]), 2 + length, nil
		}
		return "", 2 + length, nil
	case 9:
		if len(data) < 16 {
			return "", 0, io.ErrUnexpectedEOF
		}
		return "", 16, nil
	default:
		return "", 0, fmt.Errorf("bedrock: unknown event header type %d", typeID)
	}
}
