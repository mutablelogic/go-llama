package httphandler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llamacpp "github.com/mutablelogic/go-llama/pkg/llamacpp"
	schema "github.com/mutablelogic/go-llama/pkg/llamacpp/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

///////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS

// noopMiddleware returns middleware that does nothing
func noopMiddleware() HTTPMiddlewareFuncs {
	return HTTPMiddlewareFuncs{}
}

// setupTestLlama creates a Llama instance for testing
func setupTestLlama(t *testing.T) *llamacpp.Llama {
	// Create a temporary directory for test data
	tmpDir := t.TempDir()
	llama, err := llamacpp.New(tmpDir)
	require.NoError(t, err)
	return llama
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - LIST MODELS

func TestModelList_Success(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		// Clean up - unload any loaded models
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodGet, "/api/model", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusOK, rw.Code)

	var models []*schema.CachedModel
	err := json.NewDecoder(rw.Body).Decode(&models)
	assert.NoError(t, err)
	// Should return an empty list initially
	assert.NotNil(t, models)
}

func TestModelList_MethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPut, "/api/model", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - GET MODEL

func TestModelGet_EmptyID(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	// Try to get a model with empty ID - should get 400
	req := httptest.NewRequest(http.MethodGet, "/api/model/", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should get 400 for empty ID path
	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestModelGet_NonExistent(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodGet, "/api/model/nonexistent", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return 404 for nonexistent model
	assert.Equal(t, http.StatusNotFound, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - PULL MODEL

func TestModelPull_EmptyURL(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"url": ""}`
	req := httptest.NewRequest(http.MethodPost, "/api/model", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestModelPull_InvalidJSON(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPost, "/api/model", strings.NewReader("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestModelPull_MissingContentType(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"url": "hf://test/model@main/model.gguf"}`
	req := httptest.NewRequest(http.MethodPost, "/api/model", strings.NewReader(reqBody))
	// Don't set Content-Type header
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should handle missing content type gracefully
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestModelPull_StreamingAcceptHeader(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"url": "hf://microsoft/DialoGPT-medium@main/pytorch_model.bin"}`
	req := httptest.NewRequest(http.MethodPost, "/api/model", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/stream")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// The download will likely fail, but let's check the behavior
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

///////////////////////////////////////////////////////////////////////////////
// TESTS - LOAD MODEL

func TestModelLoad_EmptyName(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPost, "/api/model/", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// This should hit the pull endpoint instead
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestModelLoad_NonExistentModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPost, "/api/model/nonexistent", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return 404 for nonexistent model
	assert.Equal(t, http.StatusNotFound, rw.Code)
}

func TestModelLoad_WithRequestBody(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"contextSize": 2048}`
	req := httptest.NewRequest(http.MethodPost, "/api/model/nonexistent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return 404 because model doesn't exist
	assert.Equal(t, http.StatusNotFound, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - UNLOAD MODEL

func TestModelUnload_EmptyID(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"load": false}`
	req := httptest.NewRequest(http.MethodPost, "/api/model/", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should get 400 for empty path
	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestModelUnload_NonExistentModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"load": false}`
	req := httptest.NewRequest(http.MethodPost, "/api/model/nonexistent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return 404 for nonexistent model
	assert.Equal(t, http.StatusNotFound, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - DELETE MODEL

func TestModelDelete_EmptyID(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodDelete, "/api/model/", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should get 400 for empty path
	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestModelDelete_NonExistentModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterModelHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodDelete, "/api/model/nonexistent", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return 404 for nonexistent model
	assert.Equal(t, http.StatusNotFound, rw.Code)
}
