package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type benchmarkBackend interface {
	Generate() error
	Stream() error
	CancelLatency() (time.Duration, bool, error)
	Close() error
}

type distribution struct {
	N          int     `json:"n"`
	MeanMicros float64 `json:"mean_us"`
	P50Micros  float64 `json:"p50_us"`
	P95Micros  float64 `json:"p95_us"`
	P99Micros  float64 `json:"p99_us"`
	OpsPerSec  float64 `json:"ops_per_sec"`
	BytesPerOp uint64  `json:"allocated_bytes_per_op"`
}

type report struct {
	Backend     string       `json:"backend"`
	Commit      string       `json:"commit"`
	GoVersion   string       `json:"go_version"`
	GOOS        string       `json:"goos"`
	GOARCH      string       `json:"goarch"`
	GOMAXPROCS  int          `json:"gomaxprocs"`
	NonStream   distribution `json:"non_stream"`
	Stream      distribution `json:"stream_64_chunks"`
	Throughput  distribution `json:"throughput_16_workers"`
	RSSStartKB  int64        `json:"rss_start_kb"`
	RSSPeakKB   int64        `json:"rss_peak_kb"`
	RSSEndKB    int64        `json:"rss_end_kb"`
	CancelUS    float64      `json:"cancel_us,omitempty"`
	CancelKnown bool         `json:"cancel_supported"`
}

var (
	nonStreamN  = flag.Int("non-stream", 1500, "sequential non-stream requests")
	streamN     = flag.Int("stream", 500, "sequential 64-chunk streams")
	throughputN = flag.Int("throughput", 5000, "concurrent non-stream requests")
	workers     = flag.Int("workers", 16, "throughput workers")
	warmup      = flag.Int("warmup", 100, "warm-up requests per mode")
	procs       = flag.Int("procs", 4, "GOMAXPROCS")
)

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(*procs)
	server := newMockServer(64)
	defer server.Close()
	backend, err := newBenchmarkBackend(server.URL)
	if err != nil {
		fatal(err)
	}
	defer backend.Close()
	for range *warmup {
		if err := backend.Generate(); err != nil {
			fatal(err)
		}
	}
	for range min(*warmup, 25) {
		if err := backend.Stream(); err != nil {
			fatal(err)
		}
	}
	runtime.GC()
	startRSS := rssKB()
	stopPeak := make(chan struct{})
	peakDone := make(chan int64, 1)
	go samplePeakRSS(stopPeak, peakDone, startRSS)
	nonStream, err := measureSequential(*nonStreamN, backend.Generate)
	if err != nil {
		fatal(err)
	}
	stream, err := measureSequential(*streamN, backend.Stream)
	if err != nil {
		fatal(err)
	}
	throughput, err := measureConcurrent(*throughputN, *workers, backend.Generate)
	if err != nil {
		fatal(err)
	}
	close(stopPeak)
	peakRSS := <-peakDone
	runtime.GC()
	endRSS := rssKB()
	cancelLatency, cancelSupported, err := backend.CancelLatency()
	if err != nil {
		fatal(err)
	}
	result := report{
		Backend: backendName, Commit: backendCommit, GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		GOMAXPROCS: runtime.GOMAXPROCS(0), NonStream: nonStream, Stream: stream, Throughput: throughput,
		RSSStartKB: startRSS, RSSPeakKB: peakRSS, RSSEndKB: endRSS,
		CancelUS: float64(cancelLatency) / float64(time.Microsecond), CancelKnown: cancelSupported,
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fatal(err)
	}
	fmt.Println(string(encoded))
}

func measureSequential(n int, operation func() error) (distribution, error) {
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	latencies := make([]time.Duration, n)
	started := time.Now()
	for index := range n {
		requestStarted := time.Now()
		if err := operation(); err != nil {
			return distribution{}, err
		}
		latencies[index] = time.Since(requestStarted)
	}
	elapsed := time.Since(started)
	runtime.ReadMemStats(&after)
	result := summarize(latencies, elapsed)
	if n > 0 {
		result.BytesPerOp = (after.TotalAlloc - before.TotalAlloc) / uint64(n)
	}
	return result, nil
}

func measureConcurrent(n, concurrency int, operation func() error) (distribution, error) {
	latencies := make([]time.Duration, n)
	jobs := make(chan int)
	var workersDone sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	workersDone.Add(concurrency)
	started := time.Now()
	for range concurrency {
		go func() {
			defer workersDone.Done()
			for index := range jobs {
				requestStarted := time.Now()
				err := operation()
				latencies[index] = time.Since(requestStarted)
				if err != nil {
					errOnce.Do(func() { firstErr = err })
				}
			}
		}()
	}
	for index := range n {
		jobs <- index
	}
	close(jobs)
	workersDone.Wait()
	elapsed := time.Since(started)
	if firstErr != nil {
		return distribution{}, firstErr
	}
	return summarize(latencies, elapsed), nil
}

func summarize(latencies []time.Duration, elapsed time.Duration) distribution {
	sorted := append([]time.Duration(nil), latencies...)
	sort.Slice(sorted, func(left, right int) bool { return sorted[left] < sorted[right] })
	var total time.Duration
	for _, latency := range sorted {
		total += latency
	}
	n := len(sorted)
	if n == 0 {
		return distribution{}
	}
	micros := func(value time.Duration) float64 { return float64(value) / float64(time.Microsecond) }
	return distribution{
		N: n, MeanMicros: micros(total / time.Duration(n)), P50Micros: micros(percentile(sorted, 0.50)),
		P95Micros: micros(percentile(sorted, 0.95)), P99Micros: micros(percentile(sorted, 0.99)),
		OpsPerSec: float64(n) / elapsed.Seconds(),
	}
}

func percentile(sorted []time.Duration, quantile float64) time.Duration {
	index := int(float64(len(sorted)-1)*quantile + 0.5)
	return sorted[min(max(index, 0), len(sorted)-1)]
}

func newMockServer(chunks int) *httptest.Server {
	nonStream := []byte(`{"id":"chat-1","model":"bench","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	var stream strings.Builder
	for range chunks {
		stream.WriteString("data: {\"id\":\"chat-1\",\"model\":\"bench\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n")
	}
	stream.WriteString("data: {\"id\":\"chat-1\",\"model\":\"bench\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":64,\"total_tokens\":66}}\n\n")
	stream.WriteString("data: [DONE]\n\n")
	streamBody := stream.String()
	return httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		if request.Header.Get("X-Bench-Slow") == "1" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(response, "data: {\"id\":\"slow\",\"model\":\"bench\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n")
			if flusher, ok := response.(http.Flusher); ok {
				flusher.Flush()
			}
			<-request.Context().Done()
			return
		}
		if bytes.Contains(body, []byte(`"stream":true`)) {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, streamBody)
			return
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write(nonStream)
	}))
}

func samplePeakRSS(stop <-chan struct{}, done chan<- int64, initial int64) {
	peak := initial
	ticker := time.NewTicker(2 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			done <- peak
			return
		case <-ticker.C:
			peak = max(peak, rssKB())
		}
	}
}

func rssKB() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "VmRSS:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			value, _ := strconv.ParseInt(fields[1], 10, 64)
			return value
		}
	}
	return 0
}

func fatal(err error) {
	if !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintln(os.Stderr, err)
	}
	os.Exit(1)
}
