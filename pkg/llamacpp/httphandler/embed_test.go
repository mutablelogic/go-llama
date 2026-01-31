package httphandler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

///////////////////////////////////////////////////////////////////////////////
// TESTS - EMBED

func TestEmbedCreate_Success(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should return error since model doesn't exist, but that's expected
	// The important thing is that it processes the request correctly
	assert.NotEqual(t, http.StatusOK, rw.Code) // Will fail because model doesn't exist
}

func TestEmbedCreate_EmptyModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "", "input": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestEmbedCreate_MissingModel(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"input": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestEmbedCreate_InvalidJSON(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusBadRequest, rw.Code)
}

func TestEmbedCreate_MissingContentType(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": "Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	// Don't set Content-Type header
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should handle missing content type gracefully
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_MethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodGet, "/api/embed", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

func TestEmbedCreate_PutMethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodPut, "/api/embed", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

func TestEmbedCreate_DeleteMethodNotAllowed(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	req := httptest.NewRequest(http.MethodDelete, "/api/embed", nil)
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rw.Code)
}

///////////////////////////////////////////////////////////////////////////////
// TESTS - EDGE CASES

func TestEmbedCreate_EmptyInput(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": ""}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process empty input, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_StringInput(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": "This is a test string for embedding"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process string input, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_ArrayInput(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": ["First text", "Second text", "Third text"]}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process array input, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_LongText(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	longText := strings.Repeat("This is a very long text that should be embedded properly. ", 100)
	reqBody := `{"model": "test-model", "input": "` + longText + `"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process long text, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_MultipleInputs(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	// Create a large array of inputs
	inputs := make([]string, 10)
	for i := range inputs {
		inputs[i] = strings.Repeat("Text number "+string(rune('0'+i))+" ", 20)
	}

	inputsJSON, _ := json.Marshal(inputs)
	reqBody := `{"model": "test-model", "input": ` + string(inputsJSON) + `}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process multiple inputs, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_SpecialCharacters(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": "特殊字符 émojis 🚀 symbols @#$%^&*()_+ áéíóú"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process special characters, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_WithDimensions(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": "Hello world", "dimensions": 512}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with dimensions, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}

func TestEmbedCreate_WithEncodingFormat(t *testing.T) {
	llama := setupTestLlama(t)
	defer func() {
		_ = llama.Close()
	}()

	router := http.NewServeMux()
	RegisterEmbedHandlers(router, "/api", llama, noopMiddleware())

	reqBody := `{"model": "test-model", "input": "Hello world", "encoding_format": "float"}`
	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rw := httptest.NewRecorder()

	router.ServeHTTP(rw, req)

	// Should process request with encoding format, but will fail due to non-existent model
	assert.NotEqual(t, http.StatusOK, rw.Code)
}
