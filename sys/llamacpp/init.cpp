#include "init.h"
#include "model.h"
#include "llama.h"

#include <mutex>
#include <atomic>

///////////////////////////////////////////////////////////////////////////////
// INITIALIZATION

static std::mutex g_init_mutex;
static std::atomic<bool> g_initialized{false};

extern "C" void llama_go_init(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (!g_initialized) {
        llama_backend_init();
        g_initialized = true;
    }
}

extern "C" void llama_go_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialized) {
        // Clear model cache first
        llama_go_model_cache_clear();
        llama_backend_free();
        g_initialized = false;
    }
}

extern "C" int llama_go_is_initialized(void) {
    return g_initialized ? 1 : 0;
}
