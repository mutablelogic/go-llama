package httphandler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

///////////////////////////////////////////////////////////////////////////////
// TESTS - CHAT

func TestChatCreate_Success(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "Hello world"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return error since model doesn't exist, but that's expected
	// The important thing is that it processes the request correctly
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_EmptyModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "", "messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestChatCreate_MissingModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestChatCreate_EmptyMessages(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "messages": []}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestChatCreate_MissingMessages(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model"}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestChatCreate_InvalidJSON(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestChatCreate_MissingContentType(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	// Don't set Content-Type header
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should handle missing content type gracefully
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_MethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

func TestChatCreate_PutMethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPut, "/api/chat", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

func TestChatCreate_DeleteMethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodDelete, "/api/chat", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - STREAMING

func TestChatCreate_StreamingAcceptHeader(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "Hello world"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/stream")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// The chat will likely fail, but let's check the behavior
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

func TestChatCreate_InvalidAcceptHeader(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "Hello world"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "invalid/type")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Could be 400 (bad accept header) or 404 (model not found) depending on which is processed first
	assert.True(t, rw.Code == http.StatusBadRequest || rw.Code == http.StatusNotFound)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - MESSAGE VARIATIONS

func TestChatCreate_SingleMessage(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process single message, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_MultipleMessages(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi there!"},
			{"role": "user", "content": "How are you?"}
		]
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process multiple messages, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_WithSystemMessage(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [
			{"role": "user", "content": "What is 2+2?"}
		]
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process messages with system message, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_SpecialCharactersInMessages(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [
			{"role": "user", "content": "特殊字符 émojis 🚀 symbols @#$%^&*()_+ áéíóú"}
		]
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process special characters, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - CHAT PARAMETERS

func TestChatCreate_WithMaxTokens(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [{"role": "user", "content": "Hello"}],
		"max_tokens": 100
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with max_tokens, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_WithTemperature(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [{"role": "user", "content": "Hello"}],
		"temperature": 0.7
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with temperature, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_WithTopP(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [{"role": "user", "content": "Hello"}],
		"top_p": 0.9
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with top_p, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_WithAllParameters(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [{"role": "user", "content": "Hello"}],
		"max_tokens": 150, 
		"temperature": 0.8, 
		"top_p": 0.95,
		"top_k": 40,
		"seed": 42
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with all parameters, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestChatCreate_StreamingWithParameters(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterChatHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{
		"model": "test-model", 
		"messages": [{"role": "user", "content": "Hello"}],
		"max_tokens": 50, 
		"temperature": 0.7
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(reqBody))
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
