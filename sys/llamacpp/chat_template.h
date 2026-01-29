#ifndef LLAMA_GO_CHAT_H
#define LLAMA_GO_CHAT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Chat message structure matching llama.cpp
typedef struct {
    const char* role;     // "system", "user", "assistant"
    const char* content;  // Message content
} llama_go_chat_message;

// Note: llama_go_model_chat_template is declared in model.h

// Apply chat template to format messages
// Returns the number of bytes written (or required if buf is NULL/too small)
// Returns negative on error
int32_t llama_go_chat_apply_template(
    void* model,                          // Model handle (for default template) or NULL
    const char* tmpl,                     // Template string, or NULL to use model's default
    const llama_go_chat_message* messages,// Array of messages
    size_t n_messages,                    // Number of messages
    bool add_assistant,                   // Add assistant turn prefix at end
    char* buf,                            // Output buffer
    int32_t buf_size                      // Buffer size
);

// Get list of built-in template names
// Returns number of templates, fills output array with names
int32_t llama_go_chat_builtin_templates(const char** output, size_t max_templates);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_CHAT_H
