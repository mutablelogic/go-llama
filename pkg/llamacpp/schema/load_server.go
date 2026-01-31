//go:build !client

package schema

import (
	"sync"

	// Packages
	llamacpp "github.com/mutablelogic/go-llama/sys/llamacpp"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// ServerModel represents a model loaded in memory on the server.
// It includes the C model handle and synchronization primitives.
type ServerModel struct {
	sync.RWMutex
	Handle *llamacpp.Model
}
