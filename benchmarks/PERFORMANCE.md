# llmux Performance Baseline

English | [简体中文](#简体中文)

## English

The benchmark uses an in-process mock HTTP server with fixed requests and
responses. The environment is Linux amd64, Go 1.25.1, and `GOMAXPROCS=4`.
The host is an Apple M3 Max, with amd64 emulated by OrbStack.

Workload: 100 warm-up requests, 1,500 non-streaming requests, 500 streams with
64 chunks each, and 5,000 concurrent non-streaming requests across 16 workers.
The table reports the median of three runs.

| Metric | Median |
|---|---:|
| Non-streaming P95 | 67.375µs |
| Non-streaming P99 | 174.543µs |
| 64-chunk streaming P95 | 460.798µs |
| 64-chunk streaming P99 | 809.802µs |
| 16-worker throughput | 62,833.639 ops/s |
| Peak RSS | 18,816 KiB |
| Cancellation latency | 555.091µs |
| Azem Venat adapter overhead | +4.878% |

Absolute timings are affected by amd64 emulation and should not be treated as
native-hardware latency. See
[linux-amd64-2026-08-02.json](results/linux-amd64-2026-08-02.json) for the
three raw runs and [compare](compare/) for the executable benchmark harness.

---

## 简体中文

该基准使用进程内 mock HTTP 服务以及固定的请求和响应。测试环境为 Linux
amd64、Go 1.25.1、`GOMAXPROCS=4`；宿主机是 Apple M3 Max，amd64 由
OrbStack 模拟。

工作量包括：预热 100 次、非流式请求 1,500 次、包含 64 个分片的流式请求
500 次，以及由 16 个 worker 并发执行的 5,000 次非流式请求。下表为三轮结果的
中位数。

| 指标 | 中位数 |
|---|---:|
| 非流式 P95 | 67.375µs |
| 非流式 P99 | 174.543µs |
| 64 分片流式 P95 | 460.798µs |
| 64 分片流式 P99 | 809.802µs |
| 16 worker 吞吐量 | 62,833.639 ops/s |
| 峰值 RSS | 18,816 KiB |
| 取消延迟 | 555.091µs |
| Azem Venat 适配器开销 | +4.878% |

绝对耗时受 amd64 模拟影响，不应视为原生硬件延迟。三轮原始结果见
[linux-amd64-2026-08-02.json](results/linux-amd64-2026-08-02.json)，可执行
基准工具见 [compare](compare/)。
