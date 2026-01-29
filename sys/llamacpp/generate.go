package llamacpp

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo pkg-config: libllama go-llama
#cgo linux pkg-config: libllama-linux
#cgo darwin pkg-config: libllama-darwin
#cgo darwin CXXFLAGS: -DGGML_USE_METAL
#cgo windows pkg-config: libllama-windows
*/
import "C"

// Generate the llama pkg-config files
// Setting the prefix to the base of the repository
//go:generate go run ../pkg-config --version "0.0.0" --prefix "${PREFIX}" --cflags "-I$DOLLAR{prefix}/include" libllama.pc
//go:generate go run ../pkg-config --version "0.0.0" --prefix "${PREFIX}" --cflags "-fopenmp" --libs "-L$DOLLAR{prefix}/lib -L$DOLLAR{prefix}/lib64 -lgo-llama -lllama -lmtmd -lggml -lggml-base -lggml-cpu -lgomp -lm -lstdc++" libllama-linux.pc
//go:generate go run ../pkg-config --version "0.0.0" --prefix "${PREFIX}" --libs "-L$DOLLAR{prefix}/lib -lgo-llama -lllama -lmtmd -lggml -lggml-base -lggml-cpu -lggml-blas -lggml-metal -lstdc++ -framework Accelerate -framework Metal -framework Foundation -framework CoreGraphics" libllama-darwin.pc
//go:generate go run ../pkg-config --version "0.0.0" --prefix "${PREFIX}" --libs "-L$DOLLAR{prefix}/lib -L$DOLLAR{prefix}/lib64 -lgo-llama -lllama -lmtmd -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a -lgomp -lm -lstdc++" libllama-windows.pc
//go:generate go run ../pkg-config --version "0.0.0" --prefix "${PREFIX}" --libs "-L$DOLLAR{prefix}/lib -L$DOLLAR{prefix}/lib64 -L/usr/local/cuda/lib64 -lggml-cuda -lcudart -lcublas -lcublasLt" libllama-cuda.pc
//go:generate go run ../pkg-config --version "0.0.0" --prefix "${PREFIX}" --libs "-L$DOLLAR{prefix}/lib -L$DOLLAR{prefix}/lib64 -lggml-vulkan -lvulkan" libllama-vulkan.pc
