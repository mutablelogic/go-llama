#include "chat_template.h"
#include "model.h"
#include "error.h"
#include "llama.h"
#include <cstring>
#include <vector>
#include <string>

extern "C" {

int32_t llama_go_chat_apply_template(
    void* model,
    const char* tmpl,
    const llama_go_chat_message* messages,
    size_t n_messages,
    bool add_assistant,
    char* buf,
    int32_t buf_size
) {
    if (!messages && n_messages > 0) {
        llama_go_set_error("messages array is null");
        return -1;
    }
    
    // Get template from model if not provided
    const char* template_str = tmpl;
    if (!template_str && model) {
        template_str = llama_go_model_chat_template(model, nullptr);
    }
    
    // Convert our message format to llama_chat_message
    std::vector<llama_chat_message> llama_messages(n_messages);
    for (size_t i = 0; i < n_messages; i++) {
        llama_messages[i].role = messages[i].role;
        llama_messages[i].content = messages[i].content;
    }
    
    // Apply template
    int32_t result = llama_chat_apply_template(
        template_str,
        llama_messages.data(),
        n_messages,
        add_assistant,
        buf,
        buf_size
    );
    
    if (result < 0) {
        llama_go_set_error("failed to apply chat template");
    }
    
    return result;
}

int32_t llama_go_chat_builtin_templates(const char** output, size_t max_templates) {
    return llama_chat_builtin_templates(output, max_templates);
}

} // extern "C"
