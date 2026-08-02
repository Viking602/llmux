package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Viking602/llmux"
	"github.com/Viking602/llmux/internal/httpx"
)

type embeddingModel struct {
	provider *Provider
	id       string
}
type imageModel struct {
	provider *Provider
	id       string
}
type videoModel struct {
	provider *Provider
	id       string
}
type filesClient struct{ provider *Provider }

func (provider *Provider) EmbeddingModel(modelID string) (llmux.EmbeddingModel, error) {
	return provider.embedding(modelID)
}
func (provider *Provider) ImageModel(modelID string) (llmux.ImageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("google: model ID is empty")
	}
	return &imageModel{provider: provider, id: modelID}, nil
}
func (provider *Provider) VideoModel(modelID string) (llmux.VideoModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("google: model ID is empty")
	}
	return &videoModel{provider: provider, id: modelID}, nil
}
func (provider *Provider) Files() llmux.Files { return &filesClient{provider: provider} }
func (provider *Provider) embedding(modelID string) (*embeddingModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("google: model ID is empty")
	}
	return &embeddingModel{provider: provider, id: modelID}, nil
}
func (model *embeddingModel) ModelID() string { return model.id }
func (model *imageModel) ModelID() string     { return model.id }
func (model *videoModel) ModelID() string     { return model.id }

func (model *embeddingModel) Embed(ctx context.Context, values []string, options llmux.EmbeddingOptions) (llmux.EmbeddingResult, error) {
	if len(values) == 0 {
		return llmux.EmbeddingResult{}, errors.New("google: embedding values are empty")
	}
	modelPath := model.id
	if !strings.Contains(modelPath, "/") {
		modelPath = "models/" + modelPath
	}
	var body map[string]any
	operation := ":embedContent"
	if len(values) == 1 {
		request := map[string]any{"model": modelPath, "content": map[string]any{"parts": []any{map[string]any{"text": values[0]}}}}
		applyGoogleEmbeddingOptions(request, options)
		body = request
	} else {
		operation = ":batchEmbedContents"
		requests := make([]any, len(values))
		for index, value := range values {
			request := map[string]any{"model": modelPath, "content": map[string]any{"role": "user", "parts": []any{map[string]any{"text": value}}}}
			applyGoogleEmbeddingOptions(request, options)
			requests[index] = request
		}
		body = map[string]any{"requests": requests}
	}
	if err := mergeModalityOptions(body, options.ModelCallOptions, "content", "requests", "model"); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	response, err := modalityRequest(ctx, model.provider, http.MethodPost, model.provider.config.BaseURL+"/"+modelPath+operation, body, nil, options.Headers)
	if err != nil {
		return llmux.EmbeddingResult{}, err
	}
	defer response.Body.Close()
	var raw struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
		Embeddings []struct {
			Values []float32 `json:"values"`
		} `json:"embeddings"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64<<20)).Decode(&raw); err != nil {
		return llmux.EmbeddingResult{}, err
	}
	result := llmux.EmbeddingResult{Response: llmux.ResponseMetadata{ModelID: model.id}}
	if len(values) == 1 {
		result.Embeddings = [][]float32{raw.Embedding.Values}
	} else {
		result.Embeddings = make([][]float32, len(raw.Embeddings))
		for index := range raw.Embeddings {
			result.Embeddings[index] = raw.Embeddings[index].Values
		}
	}
	return result, nil
}

func applyGoogleEmbeddingOptions(body map[string]any, options llmux.EmbeddingOptions) {
	if options.Dimensions != nil {
		body["outputDimensionality"] = *options.Dimensions
	}
	if options.InputType != "" {
		body["taskType"] = options.InputType
	}
}

func (model *imageModel) GenerateImage(ctx context.Context, request llmux.ImageRequest) (llmux.ImageResult, error) {
	if request.Prompt == "" {
		return llmux.ImageResult{}, errors.New("google: image prompt is empty")
	}
	if strings.HasPrefix(model.id, "gemini-") || strings.Contains(model.id, "/gemini-") {
		return model.generateGeminiImage(ctx, request)
	}
	if len(request.Files) > 0 || request.Mask != nil {
		return llmux.ImageResult{}, errors.New("google: Imagen does not support image editing; use Vertex AI")
	}
	parameters := map[string]any{}
	if request.N != nil {
		parameters["sampleCount"] = *request.N
	} else {
		parameters["sampleCount"] = 1
	}
	if request.AspectRatio != "" {
		parameters["aspectRatio"] = request.AspectRatio
	}
	body := map[string]any{"instances": []any{map[string]any{"prompt": request.Prompt}}, "parameters": parameters}
	if err := mergeModalityOptions(body, request.ModelCallOptions, "instances"); err != nil {
		return llmux.ImageResult{}, err
	}
	response, err := modalityRequest(ctx, model.provider, http.MethodPost, model.provider.config.BaseURL+"/models/"+model.id+":predict", body, nil, request.Headers)
	if err != nil {
		return llmux.ImageResult{}, err
	}
	defer response.Body.Close()
	var wire struct {
		Predictions []struct {
			Bytes    string `json:"bytesBase64Encoded"`
			MimeType string `json:"mimeType"`
			Prompt   string `json:"prompt"`
		} `json:"predictions"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 256<<20)).Decode(&wire); err != nil {
		return llmux.ImageResult{}, err
	}
	result := llmux.ImageResult{Images: make([]llmux.ImageData, 0, len(wire.Predictions)), Response: llmux.ResponseMetadata{ModelID: model.id}}
	for _, prediction := range wire.Predictions {
		data, err := base64.StdEncoding.DecodeString(prediction.Bytes)
		if err != nil {
			return llmux.ImageResult{}, err
		}
		result.Images = append(result.Images, llmux.ImageData{Data: data, MediaType: prediction.MimeType, RevisedPrompt: prediction.Prompt})
	}
	if request.Size != "" {
		result.Warnings = append(result.Warnings, "size is not supported by Imagen; use aspect ratio")
	}
	if request.Seed != nil {
		result.Warnings = append(result.Warnings, "seed is not supported by this Google image endpoint")
	}
	return result, nil
}

func (model *imageModel) generateGeminiImage(ctx context.Context, request llmux.ImageRequest) (llmux.ImageResult, error) {
	if request.Mask != nil {
		return llmux.ImageResult{}, errors.New("google: Gemini image models do not support masks")
	}
	if request.N != nil && *request.N > 1 {
		return llmux.ImageResult{}, errors.New("google: Gemini image models generate one image per call")
	}
	parts := []any{map[string]any{"text": request.Prompt}}
	for _, file := range request.Files {
		if len(file.Data) == 0 {
			return llmux.ImageResult{}, errors.New("google: Gemini input images must be inline data")
		}
		parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": file.MediaType, "data": base64.StdEncoding.EncodeToString(file.Data)}})
	}
	config := map[string]any{"responseModalities": []string{"IMAGE"}}
	if request.AspectRatio != "" {
		config["imageConfig"] = map[string]any{"aspectRatio": request.AspectRatio}
	}
	if request.Seed != nil {
		config["seed"] = *request.Seed
	}
	body := map[string]any{"contents": []any{map[string]any{"role": "user", "parts": parts}}, "generationConfig": config}
	if err := mergeModalityOptions(body, request.ModelCallOptions, "contents"); err != nil {
		return llmux.ImageResult{}, err
	}
	response, err := modalityRequest(ctx, model.provider, http.MethodPost, model.provider.config.BaseURL+"/models/"+model.id+":generateContent", body, nil, request.Headers)
	if err != nil {
		return llmux.ImageResult{}, err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 256<<20))
	if err != nil {
		return llmux.ImageResult{}, err
	}
	var wire wireResponse
	if err := json.Unmarshal(payload, &wire); err != nil {
		return llmux.ImageResult{}, err
	}
	result := llmux.ImageResult{Response: llmux.ResponseMetadata{ModelID: model.id}}
	if len(wire.Candidates) > 0 {
		for _, raw := range wire.Candidates[0].Content.Parts {
			var part struct {
				InlineData *struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData"`
			}
			if json.Unmarshal(raw, &part) == nil && part.InlineData != nil {
				data, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
				if err != nil {
					return llmux.ImageResult{}, err
				}
				result.Images = append(result.Images, llmux.ImageData{Data: data, MediaType: part.InlineData.MimeType})
			}
		}
	}
	if request.Size != "" {
		result.Warnings = append(result.Warnings, "size is not supported by Gemini image models; use aspect ratio")
	}
	return result, nil
}

func (model *videoModel) GenerateVideo(ctx context.Context, request llmux.VideoRequest) (llmux.VideoResult, error) {
	if request.Prompt == "" {
		return llmux.VideoResult{}, errors.New("google: video prompt is empty")
	}
	parameters := map[string]any{}
	if request.AspectRatio != "" {
		parameters["aspectRatio"] = request.AspectRatio
	}
	if request.Resolution != "" {
		parameters["resolution"] = request.Resolution
	}
	if request.Seed != nil {
		parameters["seed"] = *request.Seed
	}
	if request.N != nil {
		parameters["sampleCount"] = *request.N
	}
	body := map[string]any{"instances": []any{map[string]any{"prompt": request.Prompt}}, "parameters": parameters}
	if err := mergeModalityOptions(body, request.ModelCallOptions, "instances"); err != nil {
		return llmux.VideoResult{}, err
	}
	response, err := modalityRequest(ctx, model.provider, http.MethodPost, model.provider.config.BaseURL+"/models/"+model.id+":predictLongRunning", body, nil, request.Headers)
	if err != nil {
		return llmux.VideoResult{}, err
	}
	var operation struct {
		Name string `json:"name"`
	}
	decodeErr := json.NewDecoder(io.LimitReader(response.Body, 4<<20)).Decode(&operation)
	_ = response.Body.Close()
	if decodeErr != nil {
		return llmux.VideoResult{}, decodeErr
	}
	if operation.Name == "" {
		return llmux.VideoResult{}, errors.New("google: video operation name is missing")
	}
	interval := 100 * time.Millisecond
	if raw := request.ProviderOptions["google"]; len(raw) > 0 {
		var opts struct {
			PollIntervalMS int `json:"pollIntervalMs"`
		}
		_ = json.Unmarshal(raw, &opts)
		if opts.PollIntervalMS > 0 {
			interval = time.Duration(opts.PollIntervalMS) * time.Millisecond
		}
	}
	for {
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return llmux.VideoResult{}, ctx.Err()
		case <-timer.C:
		}
		response, err = modalityRequest(ctx, model.provider, http.MethodGet, model.provider.config.BaseURL+"/"+strings.TrimLeft(operation.Name, "/"), nil, nil, request.Headers)
		if err != nil {
			return llmux.VideoResult{}, err
		}
		payload, readErr := io.ReadAll(io.LimitReader(response.Body, 32<<20))
		_ = response.Body.Close()
		if readErr != nil {
			return llmux.VideoResult{}, readErr
		}
		var status struct {
			Done  bool `json:"done"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
			Response struct {
				Videos []struct {
					URI      string `json:"gcsUri"`
					URL      string `json:"url"`
					Bytes    string `json:"bytesBase64Encoded"`
					MimeType string `json:"mimeType"`
				} `json:"videos"`
			} `json:"response"`
		}
		if err := json.Unmarshal(payload, &status); err != nil {
			return llmux.VideoResult{}, err
		}
		if status.Error != nil {
			return llmux.VideoResult{}, errors.New("google: " + status.Error.Message)
		}
		if !status.Done {
			continue
		}
		result := llmux.VideoResult{Response: llmux.ResponseMetadata{ModelID: model.id}}
		for _, video := range status.Response.Videos {
			item := llmux.VideoData{URL: first(video.URI, video.URL), MediaType: first(video.MimeType, "video/mp4")}
			if video.Bytes != "" {
				item.Data, err = base64.StdEncoding.DecodeString(video.Bytes)
				if err != nil {
					return llmux.VideoResult{}, err
				}
			}
			result.Videos = append(result.Videos, item)
		}
		return result, nil
	}
}

func (files *filesClient) Upload(ctx context.Context, request llmux.UploadRequest) (llmux.UploadResult, error) {
	if len(request.Data) == 0 || request.MediaType == "" {
		return llmux.UploadResult{}, errors.New("google: upload data and media type are required")
	}
	displayName := request.Filename
	interval := 2 * time.Second
	timeout := 5 * time.Minute
	if raw := request.ProviderOptions["google"]; len(raw) > 0 {
		var options struct {
			DisplayName    string `json:"displayName"`
			PollIntervalMS int    `json:"pollIntervalMs"`
			PollTimeoutMS  int    `json:"pollTimeoutMs"`
		}
		_ = json.Unmarshal(raw, &options)
		if options.DisplayName != "" {
			displayName = options.DisplayName
		}
		if options.PollIntervalMS > 0 {
			interval = time.Duration(options.PollIntervalMS) * time.Millisecond
		}
		if options.PollTimeoutMS > 0 {
			timeout = time.Duration(options.PollTimeoutMS) * time.Millisecond
		}
	}
	origin := strings.TrimSuffix(files.provider.config.BaseURL, "/v1beta")
	initHeaders := files.headers(request.Headers)
	initHeaders.Set("X-Goog-Upload-Protocol", "resumable")
	initHeaders.Set("X-Goog-Upload-Command", "start")
	initHeaders.Set("X-Goog-Upload-Header-Content-Length", fmt.Sprint(len(request.Data)))
	initHeaders.Set("X-Goog-Upload-Header-Content-Type", request.MediaType)
	initBody, _ := json.Marshal(map[string]any{"file": map[string]any{"display_name": displayName}})
	response, err := httpx.Do(ctx, files.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: origin + "/upload/v1beta/files", Headers: initHeaders, Body: initBody, Retry: files.provider.config.Retry})
	if err != nil {
		return llmux.UploadResult{}, err
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		return llmux.UploadResult{}, (&model{provider: files.provider}).responseError(response)
	}
	uploadURL := response.Header.Get("X-Goog-Upload-Url")
	_ = response.Body.Close()
	if uploadURL == "" {
		return llmux.UploadResult{}, errors.New("google: resumable upload URL is missing")
	}
	uploadHeaders := make(http.Header)
	uploadHeaders.Set("Content-Type", request.MediaType)
	uploadHeaders.Set("X-Goog-Upload-Offset", "0")
	uploadHeaders.Set("X-Goog-Upload-Command", "upload, finalize")
	response, err = httpx.Do(ctx, files.provider.config.Client, httpx.Request{Method: http.MethodPost, URL: uploadURL, Headers: uploadHeaders, Body: request.Data, Retry: files.provider.config.Retry})
	if err != nil {
		return llmux.UploadResult{}, err
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		return llmux.UploadResult{}, (&model{provider: files.provider}).responseError(response)
	}
	file, err := decodeGoogleFile(response.Body)
	_ = response.Body.Close()
	if err != nil {
		return llmux.UploadResult{}, err
	}
	deadline := time.Now().Add(timeout)
	for file.State == "PROCESSING" {
		if time.Now().After(deadline) {
			return llmux.UploadResult{}, errors.New("google: file processing timed out")
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return llmux.UploadResult{}, ctx.Err()
		case <-timer.C:
		}
		response, err = modalityRequest(ctx, files.provider, http.MethodGet, files.provider.config.BaseURL+"/"+strings.TrimLeft(file.Name, "/"), nil, nil, request.Headers)
		if err != nil {
			return llmux.UploadResult{}, err
		}
		file, err = decodeGoogleFile(response.Body)
		_ = response.Body.Close()
		if err != nil {
			return llmux.UploadResult{}, err
		}
	}
	if file.State == "FAILED" {
		return llmux.UploadResult{}, errors.New("google: file processing failed")
	}
	return llmux.UploadResult{ProviderReference: map[string]string{"name": file.Name, "uri": file.URI, "state": file.State}, MediaType: file.MimeType, Filename: first(file.DisplayName, request.Filename)}, nil
}

type googleFile struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	MimeType    string `json:"mimeType"`
	URI         string `json:"uri"`
	State       string `json:"state"`
}

func decodeGoogleFile(input io.Reader) (googleFile, error) {
	var envelope struct {
		File googleFile `json:"file"`
	}
	payload, err := io.ReadAll(io.LimitReader(input, 4<<20))
	if err != nil {
		return googleFile{}, err
	}
	if err = json.Unmarshal(payload, &envelope); err != nil {
		return googleFile{}, err
	}
	if envelope.File.Name != "" {
		return envelope.File, nil
	}
	var file googleFile
	err = json.Unmarshal(payload, &file)
	return file, err
}
func (files *filesClient) headers(overrides map[string]string) http.Header {
	return (&model{provider: files.provider}).headers(overrides)
}

func modalityRequest(ctx context.Context, provider *Provider, method, endpoint string, body map[string]any, raw []byte, overrides map[string]string) (*http.Response, error) {
	headers := (&model{provider: provider}).headers(overrides)
	payload := raw
	if body != nil {
		payload, _ = json.Marshal(body)
	}
	response, err := httpx.Do(ctx, provider.config.Client, httpx.Request{Method: method, URL: endpoint, Headers: headers, Body: payload, Retry: provider.config.Retry})
	if err != nil {
		return nil, (&model{provider: provider}).transportError(err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		return nil, (&model{provider: provider}).responseError(response)
	}
	return response, nil
}
func mergeModalityOptions(body map[string]any, options llmux.ModelCallOptions, protected ...string) error {
	blocked := make(map[string]bool)
	for _, key := range protected {
		blocked[key] = true
	}
	for _, raw := range []json.RawMessage{options.ProviderOptions["google"], options.BodyOverrides} {
		if len(raw) == 0 {
			continue
		}
		var extra map[string]any
		if json.Unmarshal(raw, &extra) != nil || extra == nil {
			return errors.New("google: modality options must be JSON objects")
		}
		for key, value := range extra {
			if blocked[key] {
				return errors.New("google: protected modality field cannot be overridden: " + key)
			}
			body[key] = value
		}
	}
	return nil
}
