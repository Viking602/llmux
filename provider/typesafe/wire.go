package typesafe

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/Viking602/llmux"
)

type question struct {
	Type         llmux.EvaluationQuestionType `json:"type"`
	Instructions json.RawMessage              `json:"instructions"`
	Criteria     json.RawMessage              `json:"criteria"`
	options      map[string]json.RawMessage
}

func buildRequest(modelID string, request llmux.EvaluationRequest) ([]byte, map[string]question, error) {
	payload, err := json.Marshal(struct {
		Model     string                              `json:"model"`
		State     any                                 `json:"state"`
		Questions map[string]llmux.EvaluationQuestion `json:"questions"`
	}{modelID, request.State, request.Questions})
	if err != nil {
		return nil, nil, fmt.Errorf("encode request: %w", err)
	}
	var wire struct {
		State     json.RawMessage     `json:"state"`
		Questions map[string]question `json:"questions"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil {
		return nil, nil, err
	}
	if !contentShape(wire.State, false) {
		return nil, nil, fmt.Errorf("state must be a JSON string, object, or array")
	}
	if len(wire.Questions) == 0 {
		return nil, nil, fmt.Errorf("at least one question is required")
	}
	for id, q := range wire.Questions {
		if err := q.validate(); err != nil {
			return nil, nil, fmt.Errorf("question %q: %w", id, err)
		}
		wire.Questions[id] = q
	}
	return payload, wire.Questions, nil
}

// All values here have already passed encoding/json validation.
func contentShape(raw json.RawMessage, nullable bool) bool {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nullable
	}
	return raw[0] == '"' || raw[0] == '{' || raw[0] == '['
}

func (q *question) validate() error {
	if !contentShape(q.Instructions, true) {
		return fmt.Errorf("instructions must be text, an object, an array, or null")
	}
	switch q.Type {
	case llmux.QuestionChoice, llmux.QuestionNoul:
		if err := json.Unmarshal(q.Criteria, &q.options); err != nil {
			if q.Type != llmux.QuestionNoul || len(q.Criteria) != 0 {
				return fmt.Errorf("criteria must be an object")
			}
		}
		if q.Type == llmux.QuestionChoice && (len(q.options) == 0 || len(q.options) > 255) {
			return fmt.Errorf("choice requires 1 to 255 options")
		}
		for key, raw := range q.options {
			if q.Type == llmux.QuestionNoul && key != "true" && key != "false" {
				return fmt.Errorf("noul criteria only accepts true and false")
			}
			if !contentShape(raw, true) {
				return fmt.Errorf("criteria descriptions must be text, objects, arrays, or null")
			}
		}
	case llmux.QuestionScore:
		var levels []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &levels); err != nil || len(levels) == 0 || len(levels) > 10 {
			return fmt.Errorf("score requires an array of 1 to 10 levels (2 or more recommended)")
		}
		q.options = make(map[string]json.RawMessage, len(levels))
		for index, raw := range levels {
			if !contentShape(raw, false) {
				return fmt.Errorf("score levels must be text, objects, or arrays")
			}
			q.options[strconv.Itoa(index)] = raw
		}
	default:
		return fmt.Errorf("unsupported question type %q", q.Type)
	}
	return nil
}

func parseResult(payload []byte, questions map[string]question) (llmux.EvaluationResult, error) {
	var wire struct {
		Model   string `json:"model"`
		Answers map[string]struct {
			llmux.EvaluationAnswer
			Probabilities map[string]*float64 `json:"probabilities"`
		} `json:"answers"`
		Usage *struct {
			InputTokens  *int `json:"input_tokens"`
			OutputTokens *int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil {
		return llmux.EvaluationResult{}, err
	}
	if strings.TrimSpace(wire.Model) == "" || len(wire.Answers) != len(questions) || wire.Usage == nil || wire.Usage.InputTokens == nil || wire.Usage.OutputTokens == nil {
		return llmux.EvaluationResult{}, fmt.Errorf("missing model, answers, or usage")
	}
	if *wire.Usage.InputTokens < 0 || *wire.Usage.OutputTokens < 0 {
		return llmux.EvaluationResult{}, fmt.Errorf("negative token usage")
	}
	result := llmux.EvaluationResult{
		Answers: make(map[string]llmux.EvaluationAnswer, len(questions)), Raw: payload,
		Response: llmux.ResponseMetadata{ModelID: wire.Model},
		Usage:    llmux.NormalizeUsage(llmux.Usage{InputTokens: *wire.Usage.InputTokens, OutputTokens: *wire.Usage.OutputTokens}),
	}
	for id, q := range questions {
		answer, ok := wire.Answers[id]
		if !ok || answer.Type != q.Type {
			return llmux.EvaluationResult{}, fmt.Errorf("answer %q is missing or has the wrong type", id)
		}
		if q.Type == llmux.QuestionNoul {
			if !unitInterval(answer.Noul) {
				return llmux.EvaluationResult{}, fmt.Errorf("answer %q has invalid noul", id)
			}
		} else {
			if !unitInterval(answer.Confidence) || len(answer.Probabilities) != len(q.options) {
				return llmux.EvaluationResult{}, fmt.Errorf("answer %q has invalid confidence or probabilities", id)
			}
			answer.EvaluationAnswer.Probabilities = make(map[string]float64, len(q.options))
			sum := 0.0
			for key := range q.options {
				probability := answer.Probabilities[key]
				if !unitInterval(probability) {
					return llmux.EvaluationResult{}, fmt.Errorf("answer %q has invalid probability for %q", id, key)
				}
				answer.EvaluationAnswer.Probabilities[key] = *probability
				sum += *probability
			}
			if math.Abs(sum-1) > 0.01 {
				return llmux.EvaluationResult{}, fmt.Errorf("answer %q probabilities do not sum to one", id)
			}
			if q.Type == llmux.QuestionChoice {
				if answer.Choice == nil {
					return llmux.EvaluationResult{}, fmt.Errorf("answer %q is missing choice", id)
				}
				if _, ok := q.options[*answer.Choice]; !ok {
					return llmux.EvaluationResult{}, fmt.Errorf("answer %q chose an unknown option", id)
				}
			} else {
				if answer.Score == nil || *answer.Score < 0 || *answer.Score > float64(len(q.options)-1) || len(answer.Legend) != len(q.options) {
					return llmux.EvaluationResult{}, fmt.Errorf("answer %q has invalid score or legend", id)
				}
				for key := range q.options {
					value, ok := answer.Legend[key]
					raw, _ := json.Marshal(value)
					if !ok || !contentShape(raw, false) {
						return llmux.EvaluationResult{}, fmt.Errorf("answer %q has invalid legend level %q", id, key)
					}
				}
			}
		}
		result.Answers[id] = answer.EvaluationAnswer
	}
	return result, nil
}

func unitInterval(value *float64) bool { return value != nil && *value >= 0 && *value <= 1 }
