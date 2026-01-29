package llamacpp

/*
#include "init.h"
*/
import "C"

///////////////////////////////////////////////////////////////////////////////
// INITIALIZATION

// Init initializes the llama.cpp backend
// This is called automatically when loading a model, but can be called
// explicitly for early initialization
// This function is idempotent - multiple calls are safe
func Init() error {
	C.llama_go_init()
	return nil
}

// Cleanup frees all resources and shuts down the backend
// This should be called when completely done with llama.cpp
func Cleanup() {
	C.llama_go_cleanup()
}

// IsInitialized returns true if the backend has been initialized
func IsInitialized() bool {
	return C.llama_go_is_initialized() != 0
}
