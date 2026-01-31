//go:build cuda

package llamacpp

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo pkg-config: libllama-cuda cuda-12.6 cublas-12.6 cudart-12.6
#cgo LDFLAGS: -L/usr/local/cuda/lib64/stubs -L/usr/local/cuda/targets/aarch64-linux/lib/stubs -L/usr/local/cuda/targets/x86_64-linux/lib/stubs
*/
import "C"
