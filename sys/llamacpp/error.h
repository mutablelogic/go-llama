#pragma once

#ifdef __cplusplus
#include <string>
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
// ERROR HANDLING
//
// Thread-local error storage for safe concurrent access from Go goroutines.

// Get the last error message (thread-local)
// Returns NULL if no error
const char* llama_go_last_error(void);

// Clear the last error
void llama_go_clear_error(void);

#ifdef __cplusplus
}

// C++ only: Set the last error message
void llama_go_set_error(const char* msg);
void llama_go_set_error(const std::string& msg);

#endif
