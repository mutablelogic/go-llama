//go:build cuda

package llamacpp

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo pkg-config: libllama-cuda cuda-12.6 cublas-12.6 cudart-12.6
*/
import "C"
