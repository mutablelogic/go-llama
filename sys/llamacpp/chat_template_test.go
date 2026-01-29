package llamacpp_test

import (
	"strings"
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelChat = "../../testdata/stories260K.gguf"

func TestBuiltinTemplates(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	templates := llamacpp.BuiltinTemplates()
	if len(templates) == 0 {
		t.Skip("No built-in templates available")
	}

	t.Logf("Found %d built-in templates:", len(templates))
	for _, tmpl := range templates {
		t.Logf("  - %s", tmpl)
	}
}

func TestApplyBuiltinTemplateChatML(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	messages := []llamacpp.ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello!"},
	}

	// Try chatml template
	result, err := llamacpp.ApplyBuiltinTemplate("chatml", messages, true)
	if err != nil {
		t.Fatalf("failed to apply chatml template: %v", err)
	}

	t.Logf("ChatML result:\n%s", result)

	// Verify it contains expected parts
	if !strings.Contains(result, "<|im_start|>") {
		t.Error("expected chatml markers in output")
	}
	if !strings.Contains(result, "system") {
		t.Error("expected system role in output")
	}
	if !strings.Contains(result, "You are a helpful assistant.") {
		t.Error("expected system content in output")
	}
}

func TestApplyBuiltinTemplateLlama2(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	messages := []llamacpp.ChatMessage{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi there"},
	}

	result, err := llamacpp.ApplyBuiltinTemplate("llama2", messages, true)
	if err != nil {
		t.Fatalf("failed to apply llama2 template: %v", err)
	}

	t.Logf("Llama2 result:\n%s", result)

	// Verify it contains expected Llama2 format
	if !strings.Contains(result, "[INST]") {
		t.Error("expected [INST] marker in output")
	}
}

func TestApplyBuiltinTemplateLlama3(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	messages := []llamacpp.ChatMessage{
		{Role: "user", Content: "What is 2+2?"},
	}

	result, err := llamacpp.ApplyBuiltinTemplate("llama3", messages, true)
	if err != nil {
		t.Fatalf("failed to apply llama3 template: %v", err)
	}

	t.Logf("Llama3 result:\n%s", result)

	// Verify it contains expected Llama3 format
	if !strings.Contains(result, "<|start_header_id|>") {
		t.Error("expected Llama3 header markers in output")
	}
}

func TestChatTemplateEmpty(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Empty messages should return empty string
	result, err := llamacpp.ApplyBuiltinTemplate("chatml", nil, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "" {
		t.Errorf("expected empty result for empty messages, got: %q", result)
	}
}

func TestModelChatTemplate(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelChat, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// stories260K may not have a chat template
	tmpl := model.ChatTemplate("")
	t.Logf("Model chat template: %q", tmpl)

	hasTmpl := model.HasChatTemplate()
	t.Logf("Has chat template: %v", hasTmpl)

	if hasTmpl && tmpl == "" {
		t.Error("HasChatTemplate returned true but template is empty")
	}
	if !hasTmpl && tmpl != "" {
		t.Error("HasChatTemplate returned false but template is not empty")
	}
}

func TestModelApplyTemplate(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelChat, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// If model has no template, this should fail gracefully
	// or we can use a built-in template
	if !model.HasChatTemplate() {
		t.Skip("Model has no chat template")
	}

	messages := []llamacpp.ChatMessage{
		{Role: "user", Content: "Hello!"},
	}

	result, err := model.ApplyTemplate(messages, true)
	if err != nil {
		t.Fatalf("failed to apply model template: %v", err)
	}

	t.Logf("Model template result:\n%s", result)
}

func TestApplyTemplateMultipleTurns(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	messages := []llamacpp.ChatMessage{
		{Role: "system", Content: "You are a math tutor."},
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "2+2 equals 4."},
		{Role: "user", Content: "And 3+3?"},
	}

	result, err := llamacpp.ApplyBuiltinTemplate("chatml", messages, true)
	if err != nil {
		t.Fatalf("failed to apply template: %v", err)
	}

	t.Logf("Multi-turn result:\n%s", result)

	// Count occurrences of role markers
	// 4 messages + 1 for the addAssistant prefix = 5 total
	count := strings.Count(result, "<|im_start|>")
	if count < 4 {
		t.Errorf("expected at least 4 message starts in output, got %d", count)
	}
}

func TestApplyTemplateNoAssistant(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	messages := []llamacpp.ChatMessage{
		{Role: "user", Content: "Hello"},
	}

	// With addAssistant = false
	result, err := llamacpp.ApplyBuiltinTemplate("chatml", messages, false)
	if err != nil {
		t.Fatalf("failed to apply template: %v", err)
	}

	t.Logf("Without assistant prefix:\n%s", result)

	// With addAssistant = true
	resultWithAss, err := llamacpp.ApplyBuiltinTemplate("chatml", messages, true)
	if err != nil {
		t.Fatalf("failed to apply template: %v", err)
	}

	t.Logf("With assistant prefix:\n%s", resultWithAss)

	// The version with assistant should be longer
	if len(resultWithAss) <= len(result) {
		t.Error("expected version with assistant prefix to be longer")
	}
}
