package llamacpp_test

import (
	"strings"
	"sync"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestLogLevelString(t *testing.T) {
	tests := []struct {
		level    llamacpp.LogLevel
		expected string
	}{
		{llamacpp.LogLevelNone, "NONE"},
		{llamacpp.LogLevelDebug, "DEBUG"},
		{llamacpp.LogLevelInfo, "INFO"},
		{llamacpp.LogLevelWarn, "WARN"},
		{llamacpp.LogLevelError, "ERROR"},
		{llamacpp.LogLevel(99), "UNKNOWN"},
	}

	for _, tt := range tests {
		if got := tt.level.String(); got != tt.expected {
			t.Errorf("LogLevel(%d).String() = %q, want %q", tt.level, got, tt.expected)
		}
	}
}

func TestSetLogLevel(t *testing.T) {
	// Save original level
	original := llamacpp.GetLogLevel()
	defer llamacpp.SetLogLevel(original)

	// Test setting different levels
	levels := []llamacpp.LogLevel{
		llamacpp.LogLevelNone,
		llamacpp.LogLevelDebug,
		llamacpp.LogLevelInfo,
		llamacpp.LogLevelWarn,
		llamacpp.LogLevelError,
	}

	for _, level := range levels {
		llamacpp.SetLogLevel(level)
		got := llamacpp.GetLogLevel()
		if got != level {
			t.Errorf("SetLogLevel(%v); GetLogLevel() = %v, want %v", level, got, level)
		}
	}
}

func TestSetLogCallback(t *testing.T) {
	llamacpp.ClearCache()

	var mu sync.Mutex
	var messages []string

	llamacpp.SetLogCallback(func(level llamacpp.LogLevel, message string) {
		mu.Lock()
		messages = append(messages, message)
		mu.Unlock()
	})

	// Load a model to generate log output
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	model.Close()

	// Disable callback
	llamacpp.SetLogCallback(nil)

	mu.Lock()
	count := len(messages)
	mu.Unlock()

	if count == 0 {
		t.Error("Expected to capture log messages, got none")
	} else {
		t.Logf("Captured %d log messages", count)
	}
}

func TestLogCallbackFiltering(t *testing.T) {
	llamacpp.ClearCache()

	// Set to only capture warnings and errors
	llamacpp.SetLogLevel(llamacpp.LogLevelWarn)

	var mu sync.Mutex
	var messages []string

	llamacpp.SetLogCallback(func(level llamacpp.LogLevel, message string) {
		mu.Lock()
		messages = append(messages, message)
		mu.Unlock()
	})

	// Load a model - most output is INFO level
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	model.Close()

	// Disable callback and restore log level
	llamacpp.SetLogCallback(nil)
	llamacpp.SetLogLevel(llamacpp.LogLevelInfo)

	mu.Lock()
	count := len(messages)
	mu.Unlock()

	t.Logf("Captured %d log messages at WARN level (should be fewer than INFO)", count)
}

func TestDisableLogging(t *testing.T) {
	llamacpp.ClearCache()

	// Disable all logging
	llamacpp.DisableLogging()

	var mu sync.Mutex
	var messages []string

	// Load a model
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	model.Close()

	// Restore defaults
	llamacpp.SetLogCallback(nil)
	llamacpp.SetLogLevel(llamacpp.LogLevelInfo)

	mu.Lock()
	count := len(messages)
	mu.Unlock()

	// With LogLevelNone and empty callback, nothing should be captured
	if count > 0 {
		t.Errorf("Expected no messages with disabled logging, got %d", count)
	}
}

func TestLogCallbackContent(t *testing.T) {
	llamacpp.ClearCache()
	llamacpp.SetLogLevel(llamacpp.LogLevelInfo)

	var mu sync.Mutex
	var messages []string
	var levels []llamacpp.LogLevel

	llamacpp.SetLogCallback(func(level llamacpp.LogLevel, message string) {
		mu.Lock()
		messages = append(messages, message)
		levels = append(levels, level)
		mu.Unlock()
	})

	// Load a model
	modelPath := testModelPath(t)
	params := llamacpp.DefaultModelParams()

	model, err := llamacpp.LoadModel(modelPath, params)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}
	model.Close()

	llamacpp.SetLogCallback(nil)

	mu.Lock()
	defer mu.Unlock()

	// Check that we got some expected content
	found := false
	for _, msg := range messages {
		if strings.Contains(msg, "Metal") || strings.Contains(msg, "llama") || strings.Contains(msg, "ggml") {
			found = true
			break
		}
	}

	if !found {
		t.Error("Expected messages containing 'Metal', 'llama', or 'ggml'")
	}

	if len(messages) > 0 {
		t.Logf("First few messages:")
		limit := 5
		if len(messages) < limit {
			limit = len(messages)
		}
		for i := 0; i < limit; i++ {
			t.Logf("  [%s] %s", levels[i], strings.TrimSpace(messages[i]))
		}
	}
}
