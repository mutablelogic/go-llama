package store

import (
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseRepoAndBranch(t *testing.T) {
	tests := []struct {
		name       string
		user       string
		repoPart   string
		wantRepo   string
		wantBranch string
	}{
		{
			name:       "repo without branch",
			user:       "microsoft",
			repoPart:   "DialoGPT-medium",
			wantRepo:   "microsoft/DialoGPT-medium",
			wantBranch: defaultBranch,
		},
		{
			name:       "repo with main branch",
			user:       "microsoft",
			repoPart:   "DialoGPT-medium@main",
			wantRepo:   "microsoft/DialoGPT-medium",
			wantBranch: "main",
		},
		{
			name:       "repo with custom branch",
			user:       "huggingface",
			repoPart:   "CodeBERTa-small-v1@dev",
			wantRepo:   "huggingface/CodeBERTa-small-v1",
			wantBranch: "dev",
		},
		{
			name:       "repo with version tag",
			user:       "openai",
			repoPart:   "whisper-large@v2.0",
			wantRepo:   "openai/whisper-large",
			wantBranch: "v2.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotRepo, gotBranch := parseRepoAndBranch(tt.user, tt.repoPart)
			assert.Equal(t, tt.wantRepo, gotRepo)
			assert.Equal(t, tt.wantBranch, gotBranch)
		})
	}
}

func TestBuildHuggingFaceURL(t *testing.T) {
	tests := []struct {
		name         string
		repo         string
		branch       string
		filePath     string
		wantURL      string
		wantDestPath string
		wantErr      bool
	}{
		{
			name:         "basic model file",
			repo:         "microsoft/DialoGPT-medium",
			branch:       "main",
			filePath:     "pytorch_model.bin",
			wantURL:      "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin?download=true",
			wantDestPath: "DialoGPT-medium/pytorch_model.bin",
			wantErr:      false,
		},
		{
			name:         "nested file path",
			repo:         "openai/whisper-large",
			branch:       "main",
			filePath:     "flax_model.msgpack",
			wantURL:      "https://huggingface.co/openai/whisper-large/resolve/main/flax_model.msgpack?download=true",
			wantDestPath: "whisper-large/flax_model.msgpack",
			wantErr:      false,
		},
		{
			name:         "deeply nested file",
			repo:         "huggingface/CodeBERTa-small-v1",
			branch:       "dev",
			filePath:     "tokenizer/vocab.txt",
			wantURL:      "https://huggingface.co/huggingface/CodeBERTa-small-v1/resolve/dev/tokenizer/vocab.txt?download=true",
			wantDestPath: "CodeBERTa-small-v1/vocab.txt",
			wantErr:      false,
		},
		{
			name:         "empty repo",
			repo:         "",
			branch:       "main",
			filePath:     "model.bin",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
		},
		{
			name:         "empty file path",
			repo:         "microsoft/DialoGPT-medium",
			branch:       "main",
			filePath:     "",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotURL, gotOpts, gotDestPath, err := buildHuggingFaceURL(tt.repo, tt.branch, tt.filePath)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, gotURL)
				assert.Empty(t, gotDestPath)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, gotURL)
			assert.Equal(t, tt.wantURL, gotURL.String())
			assert.Equal(t, tt.wantDestPath, gotDestPath)
			assert.NotEmpty(t, gotOpts)
		})
	}
}

func TestParseHFScheme(t *testing.T) {
	tests := []struct {
		name         string
		urlStr       string
		wantURL      string
		wantDestPath string
		wantErr      bool
		errContains  string
	}{
		{
			name:         "basic hf URL",
			urlStr:       "hf://microsoft/DialoGPT-medium/pytorch_model.bin",
			wantURL:      "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin?download=true",
			wantDestPath: "DialoGPT-medium/pytorch_model.bin",
			wantErr:      false,
		},
		{
			name:         "hf URL with branch",
			urlStr:       "hf://huggingface/CodeBERTa-small-v1@dev/tokenizer/vocab.txt",
			wantURL:      "https://huggingface.co/huggingface/CodeBERTa-small-v1/resolve/dev/tokenizer/vocab.txt?download=true",
			wantDestPath: "CodeBERTa-small-v1/vocab.txt",
			wantErr:      false,
		},
		{
			name:         "deeply nested path",
			urlStr:       "hf://openai/whisper-large/models/pytorch/model.safetensors",
			wantURL:      "https://huggingface.co/openai/whisper-large/resolve/main/models/pytorch/model.safetensors?download=true",
			wantDestPath: "whisper-large/model.safetensors",
			wantErr:      false,
		},
		{
			name:         "missing user",
			urlStr:       "hf:///repo/file.bin",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "missing user",
		},
		{
			name:         "missing repo",
			urlStr:       "hf://user",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "expected hf://user/repo/path/file",
		},
		{
			name:         "missing file path",
			urlStr:       "hf://user/repo",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "expected hf://user/repo/path/file",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u, err := url.Parse(tt.urlStr)
			assert.NoError(t, err)

			gotURL, gotOpts, gotDestPath, err := parseHFScheme(u)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				assert.Nil(t, gotURL)
				assert.Empty(t, gotDestPath)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, gotURL)
			assert.Equal(t, tt.wantURL, gotURL.String())
			assert.Equal(t, tt.wantDestPath, gotDestPath)
			assert.NotEmpty(t, gotOpts)
		})
	}
}

func TestParseHTTPScheme(t *testing.T) {
	tests := []struct {
		name         string
		urlStr       string
		wantURL      string
		wantDestPath string
		wantErr      bool
		errContains  string
	}{
		{
			name:         "valid HuggingFace HTTPS URL",
			urlStr:       "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin",
			wantURL:      "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin?download=true",
			wantDestPath: "DialoGPT-medium/pytorch_model.bin",
			wantErr:      false,
		},
		{
			name:         "HuggingFace URL with branch",
			urlStr:       "https://huggingface.co/openai/whisper-large/resolve/v2.0/model.safetensors",
			wantURL:      "https://huggingface.co/openai/whisper-large/resolve/v2.0/model.safetensors?download=true",
			wantDestPath: "whisper-large/model.safetensors",
			wantErr:      false,
		},
		{
			name:         "nested file path",
			urlStr:       "https://huggingface.co/huggingface/CodeBERTa-small-v1/resolve/dev/tokenizer/vocab.txt",
			wantURL:      "https://huggingface.co/huggingface/CodeBERTa-small-v1/resolve/dev/tokenizer/vocab.txt?download=true",
			wantDestPath: "CodeBERTa-small-v1/vocab.txt",
			wantErr:      false,
		},
		{
			name:         "non-HuggingFace URL",
			urlStr:       "https://example.com/model.bin",
			wantURL:      "https://example.com/model.bin",
			wantDestPath: "model.bin",
			wantErr:      false,
		},
		{
			name:         "GitHub release URL",
			urlStr:       "https://github.com/user/repo/releases/download/v1.0/model.gguf",
			wantURL:      "https://github.com/user/repo/releases/download/v1.0/model.gguf",
			wantDestPath: "model.gguf",
			wantErr:      false,
		},
		{
			name:         "invalid HuggingFace URL format",
			urlStr:       "https://huggingface.co/microsoft/DialoGPT-medium/raw/main/README.md",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "invalid huggingface.co URL format",
		},
		{
			name:         "short HuggingFace URL",
			urlStr:       "https://huggingface.co/microsoft/DialoGPT/resolve",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "invalid huggingface.co URL format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u, err := url.Parse(tt.urlStr)
			assert.NoError(t, err)

			gotURL, gotOpts, gotDestPath, err := parseHTTPScheme(u)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				assert.Nil(t, gotURL)
				assert.Empty(t, gotDestPath)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, gotURL)
			assert.Equal(t, tt.wantURL, gotURL.String())
			assert.Equal(t, tt.wantDestPath, gotDestPath)
			assert.NotEmpty(t, gotOpts)
		})
	}
}

func TestParseModelUrl(t *testing.T) {
	// Create a client for testing
	client, err := NewClient()
	assert.NoError(t, err)

	tests := []struct {
		name         string
		urlStr       string
		wantURL      string
		wantDestPath string
		wantErr      bool
		errContains  string
	}{
		{
			name:         "hf scheme URL",
			urlStr:       "hf://microsoft/DialoGPT-medium/pytorch_model.bin",
			wantURL:      "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin?download=true",
			wantDestPath: "DialoGPT-medium/pytorch_model.bin",
			wantErr:      false,
		},
		{
			name:         "https HuggingFace URL",
			urlStr:       "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin",
			wantURL:      "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin?download=true",
			wantDestPath: "DialoGPT-medium/pytorch_model.bin",
			wantErr:      false,
		},
		{
			name:         "regular https URL",
			urlStr:       "https://example.com/model.bin",
			wantURL:      "https://example.com/model.bin",
			wantDestPath: "model.bin",
			wantErr:      false,
		},
		{
			name:         "unsupported scheme",
			urlStr:       "ftp://example.com/model.bin",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "unsupported URL scheme: \"ftp\"",
		},
		{
			name:         "invalid URL",
			urlStr:       "not a url at all",
			wantURL:      "",
			wantDestPath: "",
			wantErr:      true,
			errContains:  "unsupported URL scheme",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotURL, gotOpts, gotDestPath, err := client.parseModelUrl(tt.urlStr)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				assert.Nil(t, gotURL)
				assert.Empty(t, gotDestPath)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, gotURL)
			assert.Equal(t, tt.wantURL, gotURL.String())
			assert.Equal(t, tt.wantDestPath, gotDestPath)
			assert.NotEmpty(t, gotOpts)
		})
	}
}
