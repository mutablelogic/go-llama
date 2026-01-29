#include "logging.h"
#include "llama.h"
#include "ggml.h"

#include <atomic>
#include <mutex>

///////////////////////////////////////////////////////////////////////////////
// GLOBALS

static std::atomic<llama_go_log_level> g_log_level{LLAMA_GO_LOG_LEVEL_INFO};
static std::atomic<bool> g_callback_enabled{false};
static std::mutex g_log_mutex;

///////////////////////////////////////////////////////////////////////////////
// GO CALLBACK (defined in logging.go via cgo export)

extern "C" void goLogCallback(int level, const char* text);

///////////////////////////////////////////////////////////////////////////////
// INTERNAL LOG CALLBACK

static void llama_go_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    
    // Filter by log level (CONT always passes if we're logging)
    if (level != GGML_LOG_LEVEL_CONT) {
        if (static_cast<int>(level) < static_cast<int>(g_log_level.load())) {
            return;
        }
    }
    
    // Call Go callback
    goLogCallback(static_cast<int>(level), text);
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC API

extern "C" void llama_go_log_set_level(llama_go_log_level level) {
    g_log_level.store(level);
}

extern "C" llama_go_log_level llama_go_log_get_level(void) {
    return g_log_level.load();
}

extern "C" void llama_go_log_enable_callback(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!g_callback_enabled.load()) {
        llama_log_set(llama_go_log_callback, nullptr);
        g_callback_enabled.store(true);
    }
}

extern "C" void llama_go_log_disable_callback(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_callback_enabled.load()) {
        llama_log_set(nullptr, nullptr);
        g_callback_enabled.store(false);
    }
}

extern "C" int llama_go_log_callback_enabled(void) {
    return g_callback_enabled.load() ? 1 : 0;
}
