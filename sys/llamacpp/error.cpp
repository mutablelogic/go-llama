#include "error.h"

#include <string>

///////////////////////////////////////////////////////////////////////////////
// ERROR HANDLING

// Thread-local error storage
static thread_local std::string g_last_error;

extern "C" const char* llama_go_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

extern "C" void llama_go_clear_error(void) {
    g_last_error.clear();
}

void llama_go_set_error(const char* msg) {
    g_last_error = msg ? msg : "";
}

void llama_go_set_error(const std::string& msg) {
    g_last_error = msg;
}
