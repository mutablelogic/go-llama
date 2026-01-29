package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestParseReasoningDeepSeek(t *testing.T) {
	input := `<think>
Let me think about this step by step.
First, I need to understand the question.
The user is asking about addition.
2 + 2 = 4
</think>

The answer is 4.`

	result := llamacpp.ParseReasoning(input)

	if !result.HasThinking {
		t.Error("expected HasThinking to be true")
	}

	if result.Thinking == "" {
		t.Error("expected Thinking to contain content")
	}

	t.Logf("Thinking:\n%s", result.Thinking)
	t.Logf("Content:\n%s", result.Content)

	if result.Content != "The answer is 4." {
		t.Errorf("expected Content='The answer is 4.', got %q", result.Content)
	}
}

func TestParseReasoningNoTags(t *testing.T) {
	input := "This is a simple response without any thinking tags."

	result := llamacpp.ParseReasoning(input)

	if result.HasThinking {
		t.Error("expected HasThinking to be false")
	}

	if result.Thinking != "" {
		t.Errorf("expected empty Thinking, got %q", result.Thinking)
	}

	if result.Content != input {
		t.Errorf("expected Content to equal input")
	}
}

func TestParseReasoningMultipleBlocks(t *testing.T) {
	input := `<think>First thought</think>
Some content here.
<think>Second thought</think>
More content.`

	result := llamacpp.ParseReasoning(input)

	if !result.HasThinking {
		t.Error("expected HasThinking to be true")
	}

	t.Logf("Thinking:\n%s", result.Thinking)
	t.Logf("Content:\n%s", result.Content)

	if result.Thinking == "" {
		t.Error("expected Thinking to contain both thoughts")
	}
}

func TestParseReasoningAlternativeTags(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "reasoning tag",
			input: "<reasoning>Let me reason about this.</reasoning>\nThe answer is yes.",
		},
		{
			name:  "scratchpad tag",
			input: "<scratchpad>Working out the problem...</scratchpad>\nResult: 42",
		},
		{
			name:  "thought tag",
			input: "<thought>Considering options...</thought>\nI recommend option A.",
		},
		{
			name:  "internal tag",
			input: "<internal>Processing request...</internal>\nHere is the response.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := llamacpp.ParseReasoning(tt.input)

			if !result.HasThinking {
				t.Error("expected HasThinking to be true")
			}

			if result.Thinking == "" {
				t.Error("expected Thinking to contain content")
			}

			t.Logf("Thinking: %q", result.Thinking)
			t.Logf("Content: %q", result.Content)
		})
	}
}

func TestParseReasoningWithTag(t *testing.T) {
	input := "<custom_tag>Custom thinking here</custom_tag>\nFinal answer."

	result := llamacpp.ParseReasoningWithTag(input, "custom_tag")

	if !result.HasThinking {
		t.Error("expected HasThinking to be true")
	}

	if result.Thinking != "Custom thinking here" {
		t.Errorf("expected Thinking='Custom thinking here', got %q", result.Thinking)
	}

	if result.Content != "Final answer." {
		t.Errorf("expected Content='Final answer.', got %q", result.Content)
	}
}

func TestExtractThinkingBlocks(t *testing.T) {
	input := `<think>Block 1</think>
Some text
<reasoning>Block 2</reasoning>
More text
<think>Block 3</think>`

	blocks := llamacpp.ExtractThinkingBlocks(input)

	if len(blocks) != 3 {
		t.Errorf("expected 3 blocks, got %d", len(blocks))
	}

	t.Logf("Found %d blocks:", len(blocks))
	for i, block := range blocks {
		t.Logf("  Block %d: %q", i+1, block)
	}
}

func TestStripThinking(t *testing.T) {
	input := `<think>
This is my thinking process.
I'm working through the problem.
</think>

Here is the final answer.`

	result := llamacpp.StripThinking(input)

	expected := "Here is the final answer."
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestHasThinkingTags(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"<think>content</think>", true},
		{"<reasoning>content</reasoning>", true},
		{"<scratchpad>content</scratchpad>", true},
		{"No tags here", false},
		{"<other>Not a thinking tag</other>", false},
		{"Partial <think> tag", false},
	}

	for _, tt := range tests {
		result := llamacpp.HasThinkingTags(tt.input)
		if result != tt.expected {
			t.Errorf("HasThinkingTags(%q) = %v, want %v", tt.input, result, tt.expected)
		}
	}
}

func TestParseReasoningComplex(t *testing.T) {
	input := `<think>
The user is asking about the capital of France.

Let me recall:
- France is a country in Western Europe
- Its capital city is Paris
- Paris is known for the Eiffel Tower

I should provide a clear, concise answer.
</think>

The capital of France is Paris.`

	result := llamacpp.ParseReasoning(input)

	if !result.HasThinking {
		t.Fatal("expected HasThinking to be true")
	}

	t.Logf("=== THINKING ===\n%s\n", result.Thinking)
	t.Logf("=== CONTENT ===\n%s\n", result.Content)

	if len(result.Thinking) < 50 {
		t.Error("thinking content seems too short")
	}

	if llamacpp.HasThinkingTags(result.Content) {
		t.Error("content should not contain thinking tags")
	}

	if result.Content[:3] != "The" {
		t.Errorf("content should start with 'The', got %q", result.Content[:10])
	}
}

func TestIsThinkingModel(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel("../../testdata/stories260K.gguf", modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	isThinking := model.IsThinkingModel()
	t.Logf("IsThinkingModel: %v", isThinking)
}

func TestParseReasoningEmpty(t *testing.T) {
	result := llamacpp.ParseReasoning("")

	if result.HasThinking {
		t.Error("expected HasThinking to be false for empty input")
	}

	if result.Content != "" {
		t.Error("expected empty Content for empty input")
	}
}

func TestParseReasoningOnlyThinking(t *testing.T) {
	input := "<think>Only thinking, no response</think>"

	result := llamacpp.ParseReasoning(input)

	if !result.HasThinking {
		t.Error("expected HasThinking to be true")
	}

	if result.Thinking == "" {
		t.Error("expected Thinking to have content")
	}

	if result.Content != "" {
		t.Errorf("expected empty Content, got %q", result.Content)
	}
}
