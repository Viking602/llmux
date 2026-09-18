package llmux

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
)

// EvaluationQuestion defines a closed-set judgment over shared state.
// Criteria is an object for choice/boolean and an ordered string array for score.
type EvaluationQuestion struct {
	Type         string          `json:"type"`
	Instructions string          `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

type EvaluationRequest struct {
	State     json.RawMessage               `json:"state"`
	Questions map[string]EvaluationQuestion `json:"questions"`
}

type EvaluationAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Probability   *float64           `json:"probability,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
}

type EvaluationResult struct {
	Answers map[string]EvaluationAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"inputTokens"`
		OutputTokens int `json:"outputTokens"`
	} `json:"usage"`
	Response ResponseMetadata `json:"-"`
}

type EvaluationModel interface {
	ModelID() string
	Evaluate(context.Context, EvaluationRequest) (EvaluationResult, error)
}

// Validate rejects malformed questions before any provider request is sent.
func (request EvaluationRequest) Validate() error {
	invalid := errors.New("invalid evaluation request")
	var state any
	if json.Unmarshal(request.State, &state) != nil || state == nil || len(request.Questions) == 0 {
		return invalid
	}
	switch state.(type) {
	case string, map[string]any, []any:
	default:
		return invalid
	}
	for id, question := range request.Questions {
		if strings.TrimSpace(id) == "" || strings.TrimSpace(question.Instructions) == "" {
			return invalid
		}
		switch question.Type {
		case "choice", "boolean":
			if question.Type == "boolean" && len(question.Criteria) == 0 {
				continue
			}
			var criteria map[string]string
			if json.Unmarshal(question.Criteria, &criteria) != nil || len(criteria) < 2 {
				return invalid
			}
			if question.Type == "boolean" && (len(criteria) != 2 || criteria["true"] == "" || criteria["false"] == "") {
				return invalid
			}
			for key, value := range criteria {
				if strings.TrimSpace(key) == "" || strings.TrimSpace(value) == "" {
					return invalid
				}
			}
		case "score":
			var criteria []string
			if json.Unmarshal(question.Criteria, &criteria) != nil || len(criteria) < 2 {
				return invalid
			}
			for _, value := range criteria {
				if strings.TrimSpace(value) == "" {
					return invalid
				}
			}
		default:
			return invalid
		}
	}
	return nil
}

// ValidateAnswers binds every returned answer to the exact requested question.
func (request EvaluationRequest) ValidateAnswers(result EvaluationResult) error {
	invalid := errors.New("invalid evaluation response")
	if len(result.Answers) != len(request.Questions) || result.Usage.InputTokens < 0 || result.Usage.OutputTokens < 0 {
		return invalid
	}
	for id, question := range request.Questions {
		answer, exists := result.Answers[id]
		if !exists || answer.Type != question.Type {
			return invalid
		}
		for _, probability := range answer.Probabilities {
			if !finiteRange(probability, 0, 1) {
				return invalid
			}
		}
		switch question.Type {
		case "boolean":
			if answer.Probability == nil || !finiteRange(*answer.Probability, 0, 1) {
				return invalid
			}
		case "choice":
			var criteria map[string]string
			if json.Unmarshal(question.Criteria, &criteria) != nil || criteria[answer.Choice] == "" {
				return invalid
			}
			for key := range answer.Probabilities {
				if criteria[key] == "" {
					return invalid
				}
			}
		case "score":
			var criteria []string
			if json.Unmarshal(question.Criteria, &criteria) != nil || answer.Score == nil || !finiteRange(*answer.Score, 0, float64(len(criteria)-1)) {
				return invalid
			}
		default:
			return invalid
		}
	}
	return nil
}

func finiteRange(value, lower, upper float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= lower && value <= upper
}
