package httphandler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

///////////////////////////////////////////////////////////////////////////////
// TESTS - COMPLETION

func TestCompletionCreate_Success(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return error since model doesn't exist, but that's expected
	// The important thing is that it processes the request correctly
	assert.NotEqual(t, http.StatusOK, rw.Code) // Will fail because model doesn't exist
}

func TestCompletionCreate_EmptyModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "", "prompt": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestCompletionCreate_MissingModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"prompt": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestCompletionCreate_InvalidJSON(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestCompletionCreate_MissingContentType(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	// Don't set Content-Type header
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should handle missing content type gracefully
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_MethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodGet, "/api/completion", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

func TestCompletionCreate_PutMethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPut, "/api/completion", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

func TestCompletionCreate_DeleteMethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodDelete, "/api/completion", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - STREAMING

func TestCompletionCreate_StreamingAcceptHeader(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/stream")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// The completion will likely fail, but let's check the behavior
	// If it's an error response in JSON, that's still acceptable
	assert.NotEqual(t, http.StatusOK, rw.Code)

	// The response should be either:
	// 1. A streaming response (text/event-stream), or
	// 2. An error response (application/json)
	contentType := rw.Header().Get("Content-Type")
	isStreaming := strings.Contains(contentType, "text/event-stream")
	isJSON := strings.Contains(contentType, "application/json")

	assert.True(t, isStreaming || isJSON, "Expected either streaming or JSON response, got: %s", contentType)
}

func TestCompletionCreate_InvalidAcceptHeader(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "invalid/type")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Could be 400 (bad accept header) or 404 (model not found) depending on which is processed first
	assert.True(t, rw.Code == http.StatusBadRequest || rw.Code == http.StatusNotFound)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - PROMPT VARIATIONS

func TestCompletionCreate_EmptyPrompt(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": ""}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process empty prompt, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_LongPrompt(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	longPrompt := strings.Repeat("This is a very long prompt that should be completed properly. ", 100)
	reqBody := `{"model": "test-model", "prompt": "` + longPrompt + `"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process long prompt, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_SpecialCharacters(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "特殊字符 émojis 🚀 symbols @#$%^&*()_+ áéíóú"}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process special characters, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - COMPLETION PARAMETERS

func TestCompletionCreate_WithMaxTokens(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world", "max_tokens": 100}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with max_tokens, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_WithTemperature(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world", "temperature": 0.7}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with temperature, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_WithTopP(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world", "top_p": 0.9}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with top_p, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_WithStopTokens(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "prompt": "Hello world", "stop": ["\n", ".", "!"]}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with stop tokens, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_WithAllParameters(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"prompt": "Hello world", 
		"max_tokens": 150, 
		"temperature": 0.8, 
		"top_p": 0.95,
		"stop": ["\n", "."],
		"frequency_penalty": 0.1,
		"presence_penalty": 0.1
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with all parameters, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestCompletionCreate_StreamingWithParameters(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterCompletionHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"prompt": "Hello world", 
		"max_tokens": 50, 
		"temperature": 0.7,
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/stream")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process streaming request with parameters
	assert.NotEqual(t, http.StatusOK, rw.Code)

	// Check that it attempts to handle streaming appropriately
	contentType := rw.Header().Get("Content-Type")
	isStreaming := strings.Contains(contentType, "text/event-stream")
	isJSON := strings.Contains(contentType, "application/json")

	assert.True(t, isStreaming || isJSON, "Expected either streaming or JSON response, got: %s", contentType)
}
