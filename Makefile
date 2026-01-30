# Paths to packages
DOCKER=$(shell which docker)
GIT=$(shell which git)
GO=$(shell which go)
CMAKE=$(shell which cmake)

# Set OS and Architecture
ARCH ?= $(shell arch | tr A-Z a-z | sed 's/x86_64/amd64/' | sed 's/i386/amd64/' | sed 's/armv7l/arm/' | sed 's/aarch64/arm64/')
OS ?= $(shell uname | tr A-Z a-z)
VERSION ?= $(shell git describe --tags --always | sed 's/^v//')

# Paths for building
BUILD_DIR ?= "build"
BUILD_JOBS ?= -j
PREFIX ?= ${BUILD_DIR}/install
CMAKE_FLAGS = -DBUILD_SHARED_LIBS=OFF

# Build flags
BUILD_MODULE := $(shell cat go.mod | head -1 | cut -d ' ' -f 2)
BUILD_LD_FLAGS := -X $(BUILD_MODULE)/pkg/version.GitSource=${BUILD_MODULE}
BUILD_LD_FLAGS += -X $(BUILD_MODULE)/pkg/version.GitTag=$(shell git describe --tags --always)
BUILD_LD_FLAGS += -X $(BUILD_MODULE)/pkg/version.GitBranch=$(shell git name-rev HEAD --name-only --always)
BUILD_LD_FLAGS += -X $(BUILD_MODULE)/pkg/version.GitHash=$(shell git rev-parse HEAD)
BUILD_LD_FLAGS += -X $(BUILD_MODULE)/pkg/version.GoBuildTime=$(shell date -u '+%Y-%m-%dT%H:%M:%SZ')
BUILD_FLAGS = -ldflags "-s -w $(BUILD_LD_FLAGS)" 

# If GGML_CUDA is set, then add a cuda tag for the go ${BUILD_FLAGS}
# Target specific CUDA architectures
# https://developer.nvidia.com/cuda/gpus
ifeq ($(GGML_CUDA),1)
	CMAKE_FLAGS += -DGGML_CUDA=ON
	BUILD_FLAGS += -tags cuda
	BUILD_JOBS = -j2
	ifeq ($(ARCH),arm64)
		CMAKE_FLAGS += '-DCMAKE_CUDA_ARCHITECTURES=87'
	endif
	ifeq ($(ARCH),amd64)
		CMAKE_FLAGS += '-DCMAKE_CUDA_ARCHITECTURES=75;86;89'
	endif
endif

# If GGML_VULKAN is set, then add a vulkan tag for the go ${BUILD_FLAGS}
ifeq ($(GGML_VULKAN),1)
	CMAKE_FLAGS += -DGGML_VULKAN=ON
	BUILD_FLAGS += -tags vulkan
endif

# If GGML_NATIVE is set to OFF, disable native CPU optimizations (for portable builds)
ifeq ($(GGML_NATIVE),OFF)
	CMAKE_FLAGS += -DGGML_NATIVE=OFF
endif

#####################################################################
# BUILD

# Make gollama (includes server run command)
gollama: wrapper
	@echo "Building gollama"
	@PKG_CONFIG_PATH=$(shell realpath ${PREFIX})/lib/pkgconfig CGO_LDFLAGS_ALLOW="-(W|D).*" ${GO} build ${BUILD_FLAGS} -o ${BUILD_DIR}/gollama ./cmd/gollama

#####################################################################
# BUILD STATIC LIBRARIES

libllama: cmake-dep submodule mkdir
	@echo "Building libllama for ${OS}/${ARCH}"
	@${CMAKE} -S third_party/llama.cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release ${CMAKE_FLAGS}
	@${CMAKE} --build ${BUILD_DIR} ${BUILD_JOBS} --config Release
	@${CMAKE} --install ${BUILD_DIR} --prefix $(shell realpath ${PREFIX})

# Build the Go wrapper library
wrapper: cmake-dep libllama generate
	@echo "Building go-llama wrapper"
	@PKG_CONFIG_PATH=$(shell realpath ${PREFIX})/lib/pkgconfig \
		${CMAKE} -S sys/llamacpp -B ${BUILD_DIR}/wrapper \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$(shell realpath ${PREFIX})
	@${CMAKE} --build ${BUILD_DIR}/wrapper ${BUILD_JOBS} --config Release
	@${CMAKE} --install ${BUILD_DIR}/wrapper --prefix $(shell realpath ${PREFIX})

# Generate the pkg-config files
generate: mkdir go-tidy libllama
	@echo "Generating pkg-config"
	@mkdir -p ${BUILD_DIR}/lib/pkgconfig
	@rm -f $(shell realpath ${PREFIX})/lib/pkgconfig/llama.pc
	@PKG_CONFIG_PATH=$(shell realpath ${PREFIX})/lib/pkgconfig PREFIX="$(shell realpath ${PREFIX})" go generate ./sys/llamacpp

#####################################################################
# TEST

# Test flags
TEST_FLAGS ?=

# If GGML_CUDA is set, add cuda tag for tests
ifeq ($(GGML_CUDA),1)
	TEST_FLAGS += -tags cuda
endif

# Run all tests
test: test-sys test-pkg

# Test llamacpp bindings
# Note: -p 1 is required because llama.cpp with Metal uses shared GPU state that isn't thread-safe
test-sys: wrapper
	@echo "Running tests (sys)"
	@PKG_CONFIG_PATH=$(shell realpath ${PREFIX})/lib/pkgconfig ${GO} test -p 1 ${TEST_FLAGS} ./sys/...

# Test pkg (higher-level APIs)
test-pkg:
	@echo "Running tests (pkg)"
	@PKG_CONFIG_PATH=$(shell realpath ${PREFIX})/lib/pkgconfig ${GO} test ${TEST_FLAGS} ./pkg/llamacpp/...

# Run tests with verbose output
test-v: wrapper
	@echo "Running tests (verbose)"
	@PKG_CONFIG_PATH=$(shell realpath ${PREFIX})/lib/pkgconfig ${GO} test -p 1 ${TEST_FLAGS} -v ./sys/llamacpp/...

#####################################################################
# THIRD PARTY DEPENDENCIES

.PHONY: libllama wrapper generate test test-sys test-pkg test-v submodule submodule-clean docker-dep cmake-dep git-dep go-dep mkdir clean

# Submodule checkout
submodule: git-dep
	@echo "Checking out submodules"
	@${GIT} submodule update --init --recursive --remote

# Submodule clean (ONLY cleans submodules, not main repo)
submodule-clean: git-dep
	@echo "Cleaning submodules only"
	@${GIT} submodule sync --recursive
	@${GIT} submodule update --init --force --recursive
	@${GIT} submodule foreach --recursive git clean -ffdx	

# Check for docker
docker-dep:
	@test -f "${DOCKER}" && test -x "${DOCKER}"  || (echo "Missing docker binary" && exit 1)

# Check for cmake
cmake-dep:
	@test -f "${CMAKE}" && test -x "${CMAKE}"  || (echo "Missing cmake binary" && exit 1)

# Check for git
git-dep:
	@test -f "${GIT}" && test -x "${GIT}"  || (echo "Missing git binary" && exit 1)

# Check for go
go-dep:
	@test -f "${GO}" && test -x "${GO}"  || (echo "Missing go binary" && exit 1)

#####################################################################
# CLEAN

# Make build directory
mkdir:
	@echo Mkdir ${BUILD_DIR}
	@install -d ${BUILD_DIR}
	@echo Mkdir ${PREFIX}
	@install -d ${PREFIX}

# Clean - only removes build artifacts, does NOT reset git or clean submodules
clean:
	@echo "Cleaning build artifacts"
	@rm -rf ${BUILD_DIR}

# go mod tidy
go-tidy: go-dep
	@echo Tidy
	@${GO} mod tidy
	@${GO} clean -cache
