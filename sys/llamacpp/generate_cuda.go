//go:build cuda

package whisper

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo pkg-config: libwhisper-cuda cuda-12.6 cublas-12.6 cudart-12.6
*/
import "C"
