package bedrock

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

func signV4(method string, requestURL *url.URL, body []byte, headers http.Header, credentials Credentials, region string, now time.Time) {
	if method == "" {
		method = http.MethodPost
	}
	now = now.UTC()
	date := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")
	headers.Set("Host", requestURL.Host)
	headers.Set("X-Amz-Date", amzDate)
	if credentials.SessionToken != "" {
		headers.Set("X-Amz-Security-Token", credentials.SessionToken)
	}
	keys := make([]string, 0, len(headers))
	values := make(map[string]string, len(headers))
	for key, list := range headers {
		lower := strings.ToLower(key)
		keys = append(keys, lower)
		joined := strings.Join(list, ",")
		values[lower] = strings.Join(strings.Fields(joined), " ")
	}
	sort.Strings(keys)
	keys = compact(keys)
	var canonicalHeaders strings.Builder
	for _, key := range keys {
		canonicalHeaders.WriteString(key)
		canonicalHeaders.WriteByte(':')
		canonicalHeaders.WriteString(values[key])
		canonicalHeaders.WriteByte('\n')
	}
	signedHeaders := strings.Join(keys, ";")
	payloadHash := sha256Hex(body)
	canonicalRequest := method + "\n" + requestURL.EscapedPath() + "\n" + requestURL.Query().Encode() + "\n" + canonicalHeaders.String() + "\n" + signedHeaders + "\n" + payloadHash
	scope := date + "/" + region + "/bedrock/aws4_request"
	stringToSign := "AWS4-HMAC-SHA256\n" + amzDate + "\n" + scope + "\n" + sha256Hex([]byte(canonicalRequest))
	dateKey := hmacSHA256([]byte("AWS4"+credentials.SecretAccessKey), date)
	regionKey := hmacSHA256(dateKey, region)
	serviceKey := hmacSHA256(regionKey, "bedrock")
	signingKey := hmacSHA256(serviceKey, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))
	headers.Set("Authorization", "AWS4-HMAC-SHA256 Credential="+credentials.AccessKeyID+"/"+scope+", SignedHeaders="+signedHeaders+", Signature="+signature)
	headers.Del("Host") // net/http writes Host from the URL; it remains signed.
}

func sha256Hex(value []byte) string {
	sum := sha256.Sum256(value)
	return hex.EncodeToString(sum[:])
}

func hmacSHA256(key []byte, value string) []byte {
	hash := hmac.New(sha256.New, key)
	_, _ = hash.Write([]byte(value))
	return hash.Sum(nil)
}

func compact(values []string) []string {
	if len(values) == 0 {
		return values
	}
	result := values[:1]
	for _, value := range values[1:] {
		if value != result[len(result)-1] {
			result = append(result, value)
		}
	}
	return result
}
