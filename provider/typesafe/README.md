# TypeSafe AI / Jev

TypeSafe AI is the provider (`typesafe`); Jev is its System One model. This
adapter implements native typed evaluations and model discovery using Go's
standard library and llmux's existing HTTP transport.

协议核对日期：2026-09-19。依据官方文档、官方 Python SDK 和在线 OpenAPI；下文的当前模型与服务限制是该日期的快照。

## 调用协议

| 项目 | 约定 |
| --- | --- |
| 默认 BaseURL | `https://api.typesafe.ai/v1`；自定义地址也应包含版本路径 |
| 认证 | `Authorization: Bearer <TYPESAFE_API_KEY>`；Key 由调用方传入，不自动读取或保存 |
| 评估 | `POST /systemone`，JSON 字段为 `model`、`state`、`questions` |
| 模型发现 | `GET /models`，响应为 `{"models":[{"name":...,"description":...,"release_date":...}]}` |
| 模型选择 | `jev-latest`、`jev-preview`，也接受未出现在列表中的版本 ID |
| 回包 | `answers` 按原问题 ID 对应；保留实际模型版本、输入/输出 token 和原始 JSON |

接口定义见 [API reference](https://docs.typesafe.ai/api) 和 [在线 OpenAPI](https://api.typesafe.ai/openapi.json)。
模型发现使用独立的 `models` 格式，不是 OpenAI 的 `data` 格式；当前两个别名均指向 `jev-1.13.0`，以后可能变化，实际版本应读取 `result.Response.ModelID`。[模型文档](https://docs.typesafe.ai/models)

## 三种问题

| 类型 | Criteria | 返回值 |
| --- | --- | --- |
| Choice | 选项名称到描述的 map，最多 255 项 | `Choice`、完整 `Probabilities`、`Confidence` |
| Score | 从低到高排列的描述数组，推荐 2～10 档 | 可带小数的 `Score`、每档概率、`Legend`、`Confidence` |
| Noul | 可省略，或只包含 `true` / `false` 描述的 map | `Noul`，0～1 的肯定概率 |

Score 是从 0 开始的档位加权位置，不应直接当作百分比。Noul 保留概率，由业务代码决定阈值。
问题 ID 只用于匹配响应，不会进入模型推断；问题含义应写在 `Instructions` 与选项描述中。
多个问题共享同一份 state，但各自独立评估，不能让一个问题依赖同一请求内另一个问题的答案。[Choice](https://docs.typesafe.ai/primitives/choice)、[Score](https://docs.typesafe.ai/primitives/score)、[Noul](https://docs.typesafe.ai/primitives/noul)、[问题批处理](https://docs.typesafe.ai/primitives)

`State` 支持字符串、JSON 对象或数组。`Instructions`、Choice 描述及 Noul 描述还允许 null，并可携带嵌套结构。
当前在线 OpenAPI 的 Score 允许至少 1 档，且档位描述不能为 null；本实现遵循这个请求 schema，接受 1～10 档，示例采用文档建议的多档。
这与部分说明页的“至少两档”及通用描述的 nullable 范围有差异。[结构化字段说明](https://docs.typesafe.ai/primitives/advanced)、[官方 SDK wire schema](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py)

## Go 示例

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "time"

    "github.com/Viking602/llmux"
    "github.com/Viking602/llmux/provider/typesafe"
)

func main() {
    provider, err := typesafe.New(typesafe.Config{
        APIKey: os.Getenv(typesafe.APIKeyEnvVar),
    })
    if err != nil { log.Fatal(err) }
    model, err := llmux.OpenEvaluationModel(provider, typesafe.DefaultModel)
    if err != nil { log.Fatal(err) }
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    result, err := model.Evaluate(ctx, llmux.EvaluationRequest{
        State: map[string]any{"ticket": "I was charged twice. Please refund the duplicate."},
        Questions: map[string]llmux.EvaluationQuestion{
            "team": {
                Type: llmux.QuestionChoice,
                Instructions: "Which team should handle the ticket?",
                Criteria: map[string]any{"billing": "Charges and refunds", "support": "Technical problems", "other": nil},
            },
            "urgency": {
                Type: llmux.QuestionScore,
                Instructions: "How urgent is the ticket?",
                Criteria: []string{"Can wait", "Needs attention today", "Immediate intervention"},
            },
            "duplicate_charge": {
                Type: llmux.QuestionNoul,
                Instructions: "Does the customer report being charged twice?",
            },
        },
    })
    if err != nil { log.Fatal(err) }
    fmt.Println(*result.Answers["team"].Choice)
    fmt.Println(*result.Answers["urgency"].Score)
    fmt.Println(*result.Answers["duplicate_charge"].Noul)
    fmt.Println(result.Response.ModelID, result.Usage.TotalTokens)
}
```

`provider.EvaluationModel(modelID)` 也可直接调用；`llmux.ListModels(ctx, provider)` 发现可用模型。
`catalog.Lookup("typesafe")` 与 `llmux.DescribeProvider(provider)` 均声明 `evaluation` 和模型列表能力。
`LanguageModel` 返回 `ErrorUnsupported`。Jev 不生成自由文本，也没有 token 流式输出或标准聊天工具调用；它的决策可以由上层代码用于选择工具。

## 限制、错误与验证

当前模型只接受文本形式的内容。官方公布的输入预算为整次请求 64k token、state 加最长问题 32k token；SDK 不增加 tokenizer 来猜测该限制，由服务端校验。
官方指出英语效果较好，中文任务应以实际业务样本测量准确率；发布页上的速度数据也不能作为本机、当前网络的实测结果。[模型文档](https://docs.typesafe.ai/models)

适配器复用 llmux 的连接池、取消和重试：默认最多 3 次尝试，重试 429 与 5xx（包括 529），退避受配置上限约束；`Retry.MaxAttempts: 1` 禁用重试。
HTTP 错误保留状态码和 JSON，支持 detail 字符串、对象及校验错误数组。2026-09-19 对真实 `/v1/models` 的无 Key 请求得到 HTTP 403 / `authentication_error`，因此这个明确错误类型归类为 `ErrorAuthentication`；普通 403 仍归类为权限错误。
成功响应也会校验答案 ID/类型、必填值、概率、评分范围、usage 和 16 MiB 响应上限，避免将不完整结果解释成有效决策。

本地检查：

```sh
CGO_ENABLED=0 go test ./...
```

真实认证检查（先在进程环境中配置 `TYPESAFE_API_KEY`）：

```sh
LLMUX_TYPESAFE_LIVE=1 CGO_ENABLED=0 go test ./provider/typesafe -run '^TestLiveJev$' -count=1 -v
```

该检查读取真实模型列表并执行包含三种问题的一次评估，输出实际版本与 token 计数。
未显式启用时跳过；本次开发环境未配置 Key，完成的是本地 HTTP 协议测试和无认证端点探测，尚未验证真实模型答案、延迟或效果。
