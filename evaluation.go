package llmux

import (
	"context"
	"encoding/json"
)

type EvaluationQuestionType string

const (
	QuestionChoice EvaluationQuestionType = "choice"
	QuestionScore  EvaluationQuestionType = "score"
	QuestionNoul   EvaluationQuestionType = "noul"
)

// EvaluationQuestion describes a decision rather than a text-generation prompt.
// Instructions accepts a JSON string, object, array, or nil. Criteria is an
// option-to-description map for Choice, an ordered slice for Score, or an
// optional map with "true"/"false" descriptions for Noul.
type EvaluationQuestion struct {
	Type         EvaluationQuestionType `json:"type"`
	Instructions any                    `json:"instructions"`
	Criteria     any                    `json:"criteria,omitempty"`
}

type EvaluationRequest struct {
	State     any                           `json:"state"` // JSON string, object, or array.
	Questions map[string]EvaluationQuestion `json:"questions"`
	Headers   map[string]string             `json:"headers,omitempty"`
}

// EvaluationAnswer preserves probabilities; Noul is not a thresholded boolean.
// Pointers distinguish a reported zero (or empty choice) from an absent field.
type EvaluationAnswer struct {
	Type          EvaluationQuestionType `json:"type"`
	Choice        *string                `json:"choice,omitempty"`
	Score         *float64               `json:"score,omitempty"`
	Noul          *float64               `json:"noul,omitempty"`
	Confidence    *float64               `json:"confidence,omitempty"`
	Probabilities map[string]float64     `json:"probabilities,omitempty"`
	Legend        map[string]any         `json:"legend,omitempty"`
}

type EvaluationResult struct {
	Answers  map[string]EvaluationAnswer `json:"answers"`
	Usage    Usage                       `json:"usage,omitempty"`
	Response ResponseMetadata            `json:"response,omitempty"`
	Raw      json.RawMessage             `json:"raw,omitempty"`
}

// EvaluationModel evaluates all questions against the same state in one call.
// Implementations must be safe for concurrent use.
type EvaluationModel interface {
	ModelID() string
	Evaluate(context.Context, EvaluationRequest) (EvaluationResult, error)
}

type EvaluationProvider interface {
	EvaluationModel(modelID string) (EvaluationModel, error)
}

func OpenEvaluationModel(provider Provider, modelID string) (EvaluationModel, error) {
	factory, ok := provider.(EvaluationProvider)
	if !ok {
		return nil, unsupportedProviderCapability(provider, "evaluation models")
	}
	return factory.EvaluationModel(modelID)
}
