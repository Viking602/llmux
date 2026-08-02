package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

type modalityModel struct {
	provider *Provider
	id       string
}

type filesClient struct{ provider *Provider }

func (provider *Provider) EmbeddingModel(modelID string) (llmux.EmbeddingModel, error) {
	return provider.modality(modelID)
}
func (provider *Provider) SpeechModel(modelID string) (llmux.SpeechModel, error) {
	return provider.modality(modelID)
}
func (provider *Provider) ImageModel(modelID string) (llmux.ImageModel, error) {
	return provider.modality(modelID)
}
func (provider *Provider) TranscriptionModel(modelID string) (llmux.TranscriptionModel, error) {
	return provider.modality(modelID)
}
func (provider *Provider) Files() llmux.Files { return &filesClient{provider: provider} }

func (provider *Provider) modality(modelID string) (*modalityModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("openai: model ID is empty")
	}
	return &modalityModel{provider: provider, id: modelID}, nil
}

func (model *modalityModel) ModelID() string { return model.id }

func (model *modalityModel) Embed(ctx context.Context, values []string, options llmux.EmbeddingOptions) (llmux.EmbeddingResult, error) {
	if len(values) == 0 {
		return llmux.EmbeddingResult{}, errors.New("openai: embedding values are empty")
	}
	body := map[string]any{"model": model.id, "input": values, "encoding_format": "float"}
	if options.Dimensions != nil {
		body["dimensions"] = *options.Dimensions
	}
	if err := mergeModalityOptions(body, options.ModelCallOptions, model.provider.Name(), "model", "input"); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	payload, _ := json.Marshal(body)
	response, err := model.do(ctx, "/embeddings", payload, options.Headers)
	if err != nil {
		return llmux.EmbeddingResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Model string `json:"model"`
		Data  []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64<<20)).Decode(&wire); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	sort.Slice(wire.Data, func(i, j int) bool { return wire.Data[i].Index < wire.Data[j].Index })
	result := llmux.EmbeddingResult{Embeddings: make([][]float32, len(wire.Data)), InputTokens: firstNonzero(wire.Usage.PromptTokens, wire.Usage.TotalTokens), Response: llmux.ResponseMetadata{ModelID: wire.Model}}
	for index := range wire.Data {
		result.Embeddings[index] = wire.Data[index].Embedding
	}
	return result, nil
}

func (model *modalityModel) GenerateSpeech(ctx context.Context, request llmux.SpeechRequest) (llmux.SpeechResult, error) {
	if request.Text == "" {
		return llmux.SpeechResult{}, errors.New("openai: speech text is empty")
	}
	body := map[string]any{"model": model.id, "input": request.Text, "voice": first(request.Voice, "alloy"), "response_format": first(request.OutputFormat, "mp3")}
	if request.Instructions != "" {
		body["instructions"] = request.Instructions
	}
	if request.Speed != nil {
		body["speed"] = *request.Speed
	}
	if request.Language != "" {
		body["language"] = request.Language
	}
	if err := mergeModalityOptions(body, request.ModelCallOptions, model.provider.Name(), "model", "input"); err != nil {
		return llmux.SpeechResult{}, err
	}
	payload, _ := json.Marshal(body)
	response, err := model.do(ctx, "/audio/speech", payload, request.Headers)
	if err != nil {
		return llmux.SpeechResult{}, err
	}
	defer response.Body.Close()
	audio, err := io.ReadAll(io.LimitReader(response.Body, 256<<20))
	if err != nil {
		return llmux.SpeechResult{}, err
	}
	return llmux.SpeechResult{Audio: audio, MediaType: response.Header.Get("Content-Type"), Response: llmux.ResponseMetadata{ModelID: model.id}}, nil
}

func (model *modalityModel) GenerateImage(ctx context.Context, request llmux.ImageRequest) (llmux.ImageResult, error) {
	if request.Prompt == "" {
		return llmux.ImageResult{}, errors.New("openai: image prompt is empty")
	}
	var response *http.Response
	var err error
	if len(request.Files) == 0 && request.Mask == nil {
		body := map[string]any{"model": model.id, "prompt": request.Prompt, "response_format": "b64_json"}
		if request.N != nil {
			body["n"] = *request.N
		}
		if request.Size != "" {
			body["size"] = request.Size
		}
		if request.AspectRatio != "" {
			body["aspect_ratio"] = request.AspectRatio
		}
		if err = mergeModalityOptions(body, request.ModelCallOptions, model.provider.Name(), "model", "prompt"); err != nil {
			return llmux.ImageResult{}, err
		}
		payload, _ := json.Marshal(body)
		response, err = model.do(ctx, "/images/generations", payload, request.Headers)
	} else {
		var buffer bytes.Buffer
		writer := multipart.NewWriter(&buffer)
		_ = writer.WriteField("model", model.id)
		_ = writer.WriteField("prompt", request.Prompt)
		_ = writer.WriteField("response_format", "b64_json")
		if request.N != nil {
			_ = writer.WriteField("n", strconv.Itoa(*request.N))
		}
		if request.Size != "" {
			_ = writer.WriteField("size", request.Size)
		}
		for index, file := range request.Files {
			name := first(file.Filename, fmt.Sprintf("image-%d", index))
			if err = writeFile(writer, "image[]", name, file.Data); err != nil {
				return llmux.ImageResult{}, err
			}
		}
		if request.Mask != nil {
			if err = writeFile(writer, "mask", first(request.Mask.Filename, "mask.png"), request.Mask.Data); err != nil {
				return llmux.ImageResult{}, err
			}
		}
		if err = addMultipartOptions(writer, request.ModelCallOptions, model.provider.Name()); err != nil {
			return llmux.ImageResult{}, err
		}
		_ = writer.Close()
		response, err = model.doMultipart(ctx, "/images/edits", buffer.Bytes(), writer.FormDataContentType(), request.Headers)
	}
	if err != nil {
		return llmux.ImageResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Data []struct {
			B64JSON       string `json:"b64_json"`
			URL           string `json:"url"`
			RevisedPrompt string `json:"revised_prompt"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 256<<20)).Decode(&wire); err != nil {
		return llmux.ImageResult{}, err
	}
	result := llmux.ImageResult{Images: make([]llmux.ImageData, 0, len(wire.Data)), Response: llmux.ResponseMetadata{ModelID: model.id}}
	for _, item := range wire.Data {
		image := llmux.ImageData{URL: item.URL, RevisedPrompt: item.RevisedPrompt}
		if item.B64JSON != "" {
			image.Data, err = base64.StdEncoding.DecodeString(item.B64JSON)
			if err != nil {
				return llmux.ImageResult{}, err
			}
		}
		result.Images = append(result.Images, image)
	}
	return result, nil
}

func (model *modalityModel) Transcribe(ctx context.Context, request llmux.TranscriptionRequest) (llmux.TranscriptionResult, error) {
	if len(request.Audio) == 0 {
		return llmux.TranscriptionResult{}, errors.New("openai: transcription audio is empty")
	}
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	filename := first(request.Filename, "audio"+extension(request.MediaType))
	if err := writeFile(writer, "file", filename, request.Audio); err != nil {
		return llmux.TranscriptionResult{}, err
	}
	_ = writer.WriteField("model", model.id)
	_ = writer.WriteField("response_format", first(request.Format, "verbose_json"))
	if request.Language != "" {
		_ = writer.WriteField("language", request.Language)
	}
	if request.Prompt != "" {
		_ = writer.WriteField("prompt", request.Prompt)
	}
	if request.Temperature != nil {
		_ = writer.WriteField("temperature", strconv.FormatFloat(*request.Temperature, 'g', -1, 64))
	}
	if err := addMultipartOptions(writer, request.ModelCallOptions, model.provider.Name()); err != nil {
		return llmux.TranscriptionResult{}, err
	}
	_ = writer.Close()
	response, err := model.doMultipart(ctx, "/audio/transcriptions", buffer.Bytes(), writer.FormDataContentType(), request.Headers)
	if err != nil {
		return llmux.TranscriptionResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Text     string  `json:"text"`
		Language string  `json:"language"`
		Duration float64 `json:"duration"`
		Segments []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"segments"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&wire); err != nil {
		return llmux.TranscriptionResult{}, err
	}
	result := llmux.TranscriptionResult{Text: wire.Text, Language: wire.Language, DurationSeconds: wire.Duration, Response: llmux.ResponseMetadata{ModelID: model.id}, Segments: make([]llmux.TranscriptionSegment, len(wire.Segments))}
	for index, segment := range wire.Segments {
		result.Segments[index] = llmux.TranscriptionSegment{Text: segment.Text, Start: segment.Start, End: segment.End}
	}
	return result, nil
}

func (files *filesClient) Upload(ctx context.Context, request llmux.UploadRequest) (llmux.UploadResult, error) {
	if len(request.Data) == 0 {
		return llmux.UploadResult{}, errors.New("openai: upload data is empty")
	}
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	if err := writeFile(writer, "file", first(request.Filename, "upload"), request.Data); err != nil {
		return llmux.UploadResult{}, err
	}
	_ = writer.WriteField("purpose", first(request.Purpose, "assistants"))
	if err := addMultipartOptions(writer, request.ModelCallOptions, files.provider.Name()); err != nil {
		return llmux.UploadResult{}, err
	}
	_ = writer.Close()
	model := &modalityModel{provider: files.provider}
	response, err := model.doMultipart(ctx, "/files", buffer.Bytes(), writer.FormDataContentType(), request.Headers)
	if err != nil {
		return llmux.UploadResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		ID       string `json:"id"`
		Object   string `json:"object"`
		Filename string `json:"filename"`
		Purpose  string `json:"purpose"`
		Status   string `json:"status"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 4<<20)).Decode(&wire); err != nil {
		return llmux.UploadResult{}, err
	}
	return llmux.UploadResult{ProviderReference: map[string]string{"id": wire.ID, "object": wire.Object, "purpose": wire.Purpose, "status": wire.Status}, MediaType: request.MediaType, Filename: wire.Filename}, nil
}

func (model *modalityModel) do(ctx context.Context, path string, body []byte, overrides map[string]string) (*http.Response, error) {
	return model.request(ctx, path, body, "application/json", overrides)
}
func (model *modalityModel) doMultipart(ctx context.Context, path string, body []byte, contentType string, overrides map[string]string) (*http.Response, error) {
	return model.request(ctx, path, body, contentType, overrides)
}

func (model *modalityModel) request(ctx context.Context, path string, body []byte, contentType string, overrides map[string]string) (*http.Response, error) {
	base := &modelForHeaders{provider: model.provider}
	headers := base.headers(overrides)
	headers.Set("Content-Type", contentType)
	response, err := httpx.Do(ctx, model.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: model.provider.config.BaseURL + path, Headers: headers, Body: body, Retry: model.provider.config.Retry})
	if err != nil {
		return nil, base.transportError(err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		return nil, base.responseError(response)
	}
	return response, nil
}

type modelForHeaders = model

func mergeModalityOptions(body map[string]any, options llmux.ModelCallOptions, provider string, protected ...string) error {
	blocked := make(map[string]bool, len(protected))
	for _, key := range protected {
		blocked[key] = true
	}
	for _, raw := range []json.RawMessage{options.ProviderOptions[provider], options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if err := json.Unmarshal(raw, &extra); err != nil || extra == nil {
			return errors.New("openai: modality options must be JSON objects")
		}
		for key, value := range extra {
			if blocked[key] {
				return fmt.Errorf("openai: %q cannot be overridden", key)
			}
			body[key] = value
		}
	}
	return nil
}

func addMultipartOptions(writer *multipart.Writer, options llmux.ModelCallOptions, provider string) error {
	for _, raw := range []json.RawMessage{options.ProviderOptions[provider], options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if err := json.Unmarshal(raw, &extra); err != nil || extra == nil {
			return errors.New("openai: multipart options must be JSON objects")
		}
		for key, value := range extra {
			if key == "model" || key == "file" {
				return fmt.Errorf("openai: %q cannot be overridden", key)
			}
			encoded, _ := json.Marshal(value)
			if text, ok := value.(string); ok {
				_ = writer.WriteField(key, text)
			} else {
				_ = writer.WriteField(key, string(encoded))
			}
		}
	}
	return nil
}
func writeFile(writer *multipart.Writer, field, filename string, data []byte) error {
	part, err := writer.CreateFormFile(field, filename)
	if err != nil {
		return err
	}
	_, err = part.Write(data)
	return err
}
func extension(mediaType string) string {
	return map[string]string{"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm", "audio/mp4": ".m4a", "audio/ogg": ".ogg"}[mediaType]
}
func firstNonzero(values ...int) int {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}
