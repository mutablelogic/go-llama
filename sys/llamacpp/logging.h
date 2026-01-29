#ifndef LLAMA_GO_LOGGING_H
#define LLAMA_GO_LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

// Log levels matching ggml
typedef enum {
    LLAMA_GO_LOG_LEVEL_NONE  = 0,
    LLAMA_GO_LOG_LEVEL_DEBUG = 1,
    LLAMA_GO_LOG_LEVEL_INFO  = 2,
    LLAMA_GO_LOG_LEVEL_WARN  = 3,
    LLAMA_GO_LOG_LEVEL_ERROR = 4,
} llama_go_log_level;

// Set the minimum log level (messages below this level are discarded)
void llama_go_log_set_level(llama_go_log_level level);

// Get the current log level
llama_go_log_level llama_go_log_get_level(void);

// Enable the Go log callback (routes logs to Go)
void llama_go_log_enable_callback(void);

// Disable the Go log callback (restores default stderr logging)
void llama_go_log_disable_callback(void);

// Check if Go callback is enabled
int llama_go_log_callback_enabled(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_GO_LOGGING_H
