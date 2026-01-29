#ifndef GO_LLAMA_TOKENIZER_H
#define GO_LLAMA_TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Tokenize text into tokens
// Returns number of tokens, or negative on error
// If tokens is NULL, returns required buffer size
int32_t llama_go_tokenize(
    void* model,
    const char* text,
    int32_t text_len,
    int32_t* tokens,
    int32_t tokens_capacity,
    bool add_special,
    bool parse_special
);

// Convert a single token to text piece
// Returns the length of the piece, or negative on error
int32_t llama_go_token_to_piece(
    void* model,
    int32_t token,
    char* buf,
    int32_t buf_size,
    bool special
);

// Decode tokens to text
// Returns length of decoded text, or negative on error
int32_t llama_go_detokenize(
    void* model,
    const int32_t* tokens,
    int32_t n_tokens,
    char* text,
    int32_t text_capacity,
    bool remove_special,
    bool unparse_special
);

// Special tokens
int32_t llama_go_token_bos(void* model);
int32_t llama_go_token_eos(void* model);
int32_t llama_go_token_eot(void* model);
int32_t llama_go_token_nl(void* model);
int32_t llama_go_token_pad(void* model);

// Check if token is special
bool llama_go_token_is_eog(void* model, int32_t token);
bool llama_go_token_is_control(void* model, int32_t token);

// Vocabulary info
int32_t llama_go_n_vocab(void* model);

#ifdef __cplusplus
}
#endif

#endif // GO_LLAMA_TOKENIZER_H
