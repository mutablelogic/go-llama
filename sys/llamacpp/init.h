#pragma once

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
// INITIALIZATION
//
// Global llama.cpp backend lifecycle management.

// Initialize llama backend (call once at startup)
// Safe to call multiple times; only initializes once
void llama_go_init(void);

// Cleanup llama backend (call once at shutdown)
// Clears model cache and frees backend resources
void llama_go_cleanup(void);

// Check if backend is initialized
int llama_go_is_initialized(void);

#ifdef __cplusplus
}
#endif
