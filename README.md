# llmux

English | [简体中文](#简体中文)

## English

`llmux` is a unified AI provider SDK written in pure Go.

Implemented:

- Native adapters for OpenAI, Anthropic, Google, Vertex AI, Amazon Bedrock, Azure OpenAI, Cohere, Mistral, xAI, DeepSeek, OpenResponses, Voyage AI, and Tavily
- A generic provider registry that prefers the Responses API when available, then Anthropic Messages, and finally Chat Completions
- Text generation, reasoning, tool calling, embeddings, reranking, speech, transcription, image, video, file, and search interfaces
- Bounded SSE parsing, connection reuse, exponential backoff, `Retry-After`, and context cancellation

### Quick start

```go
provider, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
if err != nil {
    log.Fatal(err)
}
model, err := provider.LanguageModel("gpt-5.6")
if err != nil {
    log.Fatal(err)
}
result, err := model.Generate(ctx, llmux.Request{
    Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")},
})
```

### Verification

```bash
CGO_ENABLED=0 go test ./...
go test -run '^$' -bench . -benchmem ./benchmarks
```

---

## 简体中文

`llmux` 是纯 Go 实现的统一 AI 供应商 SDK。

当前已实现：

- OpenAI、Anthropic、Google、Vertex AI、Amazon Bedrock、Azure OpenAI、Cohere、Mistral、xAI、DeepSeek、OpenResponses、Voyage AI 与 Tavily 原生适配
- 通用供应商注册表：优先使用供应商提供的 Responses API，其次使用 Anthropic Messages，最后回退到 Chat Completions
- 文本生成、推理、工具调用、向量嵌入、重排序、语音、转录、图片、视频、文件与搜索接口
- 有界 SSE 解析、连接复用、指数退避、`Retry-After` 与上下文取消

### 快速开始

```go
provider, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
if err != nil {
    log.Fatal(err)
}
model, err := provider.LanguageModel("gpt-5.6")
if err != nil {
    log.Fatal(err)
}
result, err := model.Generate(ctx, llmux.Request{
    Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")},
})
```

### 验证

```bash
CGO_ENABLED=0 go test ./...
go test -run '^$' -bench . -benchmem ./benchmarks
```
