# llmux

English | [简体中文](#简体中文)

## English

`llmux` is a unified AI provider SDK written in pure Go.

Implemented:

- Native adapters for OpenAI, OpenAI Codex, Anthropic, Google, Vertex AI, Amazon Bedrock, Azure OpenAI, Cohere, Mistral, xAI, DeepSeek, OpenResponses, Voyage AI, and Tavily
- A generic provider registry, including InferenceHub, that prefers the Responses API when available, then Anthropic Messages, and finally Chat Completions
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
maxOutput := 8192
result, err := model.Generate(ctx, llmux.Request{
    Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")},
    Options:  llmux.CallOptions{MaxOutputTokens: &maxOutput},
})
```

Anthropic Messages (including Anthropic-compatible vendors such as DeepSeek) requires `max_tokens`. Set `CallOptions.MaxOutputTokens` on each request, or set `anthropic.Config.DefaultMaxOutputTokens` / `compat.Config.DefaultMaxOutputTokens` for a provider-level default. When neither is set, the library uses `4096`.

### Codex and InferenceHub

```go
codexProvider, err := codex.New(codex.Config{
    APIKey: os.Getenv(codex.APIKeyEnvVar),
})
subscriptionProvider, err := codex.New(codex.Config{
    APIKey: accountToken,
    Mode: codex.ModeSubscription,
    ChatGPTAccountID: accountID,
})
inferenceHub, err := compat.New("inferencehub", compat.Config{
    APIKey: os.Getenv("INFERENCEHUB_API_KEY"),
})
```

The ChatGPT subscription endpoint is undocumented and best-effort. The caller owns OAuth login, token storage, and the refresh-and-retry loop. `codex.Refresh` performs one non-retried refresh-token exchange.

### Verification

```bash
CGO_ENABLED=0 go test ./...
go test -run '^$' -bench . -benchmem ./benchmarks
```

### Acknowledgements

Thanks to the [AIMux](https://github.com/arcships/aimux) project for its provider research and protocol references.

---

## 简体中文

`llmux` 是纯 Go 实现的统一 AI 供应商 SDK。

当前已实现：

- OpenAI、OpenAI Codex、Anthropic、Google、Vertex AI、Amazon Bedrock、Azure OpenAI、Cohere、Mistral、xAI、DeepSeek、OpenResponses、Voyage AI 与 Tavily 原生适配
- 通用供应商注册表（包含 InferenceHub）：优先使用供应商提供的 Responses API，其次使用 Anthropic Messages，最后回退到 Chat Completions
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
maxOutput := 8192
result, err := model.Generate(ctx, llmux.Request{
    Messages: []llmux.Message{llmux.TextMessage(llmux.RoleUser, "hello")},
    Options:  llmux.CallOptions{MaxOutputTokens: &maxOutput},
})
```

Anthropic Messages（含 DeepSeek 等 Anthropic 兼容供应商）必须提供 `max_tokens`。请在请求上设置 `CallOptions.MaxOutputTokens`，或在 `anthropic.Config.DefaultMaxOutputTokens` / `compat.Config.DefaultMaxOutputTokens` 配置提供商级默认值。两者都未设置时，库使用 `4096`。

### Codex 与 InferenceHub

```go
codexProvider, err := codex.New(codex.Config{
    APIKey: os.Getenv(codex.APIKeyEnvVar),
})
subscriptionProvider, err := codex.New(codex.Config{
    APIKey: accountToken,
    Mode: codex.ModeSubscription,
    ChatGPTAccountID: accountID,
})
inferenceHub, err := compat.New("inferencehub", compat.Config{
    APIKey: os.Getenv("INFERENCEHUB_API_KEY"),
})
```

ChatGPT 订阅端点属于未公开、尽力而为的接入方式。调用方负责 OAuth 登录、令牌持久化以及刷新后重试；`codex.Refresh` 只执行一次不重试的刷新令牌交换。

### 验证

```bash
CGO_ENABLED=0 go test ./...
go test -run '^$' -bench . -benchmem ./benchmarks
```

### 致谢

感谢 [AIMux](https://github.com/arcships/aimux) 项目提供的供应商研究和协议实现参考。
