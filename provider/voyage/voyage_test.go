package voyage

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Viking602/llmux"
)

func TestEmbeddingsAreSortedAndRerankParses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/embeddings" {
			_, _ = response.Write([]byte(`{"data":[{"index":1,"embedding":[2]},{"index":0,"embedding":[1]}],"usage":{"total_tokens":4}}`))
			return
		}
		_, _ = response.Write([]byte(`{"data":[{"index":1,"relevance_score":0.9}]}`))
	}))
	defer server.Close()
	provider, err := New(Config{APIKey: "key", BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	embedding, _ := provider.EmbeddingModel("voyage-test")
	embedded, err := embedding.Embed(context.Background(), []string{"a", "b"}, llmux.EmbeddingOptions{})
	if err != nil || embedded.Embeddings[0][0] != 1 || embedded.InputTokens != 4 {
		t.Fatalf("embedding/error = %#v/%v", embedded, err)
	}
	reranker, _ := provider.RerankingModel("rerank-test")
	ranked, err := reranker.Rerank(context.Background(), "q", []llmux.RerankDocument{{Text: "a"}, {Text: "b"}}, llmux.RerankOptions{})
	if err != nil || len(ranked.Ranking) != 1 || ranked.Ranking[0].Index != 1 {
		t.Fatalf("ranking/error = %#v/%v", ranked, err)
	}
}
