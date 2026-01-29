package llamacpp

/*
#include "logging.h"
#include <stdlib.h>
*/
import "C"
import (
	"sync"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// LOG LEVELS

// LogLevel represents the severity of a log message
type LogLevel int

const (
	LogLevelNone  LogLevel = C.LLAMA_GO_LOG_LEVEL_NONE
	LogLevelDebug LogLevel = C.LLAMA_GO_LOG_LEVEL_DEBUG
	LogLevelInfo  LogLevel = C.LLAMA_GO_LOG_LEVEL_INFO
	LogLevelWarn  LogLevel = C.LLAMA_GO_LOG_LEVEL_WARN
	LogLevelError LogLevel = C.LLAMA_GO_LOG_LEVEL_ERROR
)

// String returns the log level name
func (l LogLevel) String() string {
	switch l {
	case LogLevelNone:
		return "NONE"
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

///////////////////////////////////////////////////////////////////////////////
// LOG CALLBACK

// LogCallback is a function that receives log messages
type LogCallback func(level LogLevel, message string)

var (
	logCallbackMu   sync.RWMutex
	logCallbackFunc LogCallback
)

// SetLogCallback sets a custom log callback function.
// Pass nil to disable custom logging and restore default behavior.
// The callback receives the log level and message text.
func SetLogCallback(fn LogCallback) {
	logCallbackMu.Lock()
	defer logCallbackMu.Unlock()

	logCallbackFunc = fn
	if fn != nil {
		C.llama_go_log_enable_callback()
	} else {
		C.llama_go_log_disable_callback()
	}
}

// SetLogLevel sets the minimum log level.
// Messages below this level will be discarded.
func SetLogLevel(level LogLevel) {
	C.llama_go_log_set_level(C.llama_go_log_level(level))
}

// GetLogLevel returns the current minimum log level.
func GetLogLevel() LogLevel {
	return LogLevel(C.llama_go_log_get_level())
}

// DisableLogging suppresses all log output from llama.cpp
func DisableLogging() {
	SetLogLevel(LogLevelNone)
	SetLogCallback(func(LogLevel, string) {})
}

///////////////////////////////////////////////////////////////////////////////
// CGO EXPORT (called from C++)

//export goLogCallback
func goLogCallback(level C.int, text *C.char) {
	logCallbackMu.RLock()
	fn := logCallbackFunc
	logCallbackMu.RUnlock()

	if fn != nil {
		fn(LogLevel(level), C.GoString(text))
	}
}

// Ensure the export is not optimized away
var _ = unsafe.Pointer(nil)
