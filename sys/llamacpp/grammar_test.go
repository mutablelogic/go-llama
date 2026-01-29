package llamacpp

import (
	"os"
	"testing"
)

const testModelGrammar = "../../testdata/stories260K.gguf"

func TestGrammarSampler(t *testing.T) {
	// Skip if model doesn't exist
	if _, err := os.Stat(testModelGrammar); os.IsNotExist(err) {
		t.Skipf("Skipping test: model not found at %s", testModelGrammar)
	}

	Init()
	defer Cleanup()

	model, err := LoadModel(testModelGrammar, DefaultModelParams())
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	// Grammar that only allows "Yes" or "No"
	grammar := `root ::= "Yes" | "No"`

	sampler, err := NewGrammarSampler(model, grammar, "root")
	if err != nil {
		t.Fatal(err)
	}
	defer sampler.Close()

	t.Logf("Grammar sampler created successfully")
}

func TestGrammarSampler_JSON(t *testing.T) {
	if _, err := os.Stat(testModelGrammar); os.IsNotExist(err) {
		t.Skipf("Skipping test: model not found at %s", testModelGrammar)
	}

	Init()
	defer Cleanup()

	model, err := LoadModel(testModelGrammar, DefaultModelParams())
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	// Simple JSON grammar
	grammar := `
root ::= object
object ::= "{" ws "\"name\"" ws ":" ws string ws "}"
string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
`

	sampler, err := NewGrammarSampler(model, grammar, "root")
	if err != nil {
		t.Fatal(err)
	}
	defer sampler.Close()

	t.Logf("JSON grammar sampler created successfully")
}

func TestLazyGrammarSampler(t *testing.T) {
	if _, err := os.Stat(testModelGrammar); os.IsNotExist(err) {
		t.Skipf("Skipping test: model not found at %s", testModelGrammar)
	}

	Init()
	defer Cleanup()

	model, err := LoadModel(testModelGrammar, DefaultModelParams())
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	// Grammar for simple object
	grammar := `
root ::= object
object ::= "{" ws "\"result\"" ws ":" ws string ws "}"
string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
`

	// Create lazy sampler that triggers on "{"
	opts := LazyGrammarOptions{
		GrammarStr:      grammar,
		GrammarRoot:     "root",
		TriggerPatterns: []string{`\{`},
	}

	sampler, err := NewLazyGrammarSampler(model, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer sampler.Close()

	t.Logf("Lazy grammar sampler created successfully")
}

func TestGrammarSampler_InvalidGrammar(t *testing.T) {
	if _, err := os.Stat(testModelGrammar); os.IsNotExist(err) {
		t.Skipf("Skipping test: model not found at %s", testModelGrammar)
	}

	Init()
	defer Cleanup()

	model, err := LoadModel(testModelGrammar, DefaultModelParams())
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	// Invalid grammar syntax
	invalidGrammar := `root ::= [ this is invalid`

	_, err = NewGrammarSampler(model, invalidGrammar, "root")
	if err == nil {
		t.Error("Expected error for invalid grammar")
	}
	t.Logf("Got expected error: %v", err)
}

func TestGrammarSampler_EmptyGrammar(t *testing.T) {
	if _, err := os.Stat(testModelGrammar); os.IsNotExist(err) {
		t.Skipf("Skipping test: model not found at %s", testModelGrammar)
	}

	Init()
	defer Cleanup()

	model, err := LoadModel(testModelGrammar, DefaultModelParams())
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	_, err = NewGrammarSampler(model, "", "root")
	if err == nil {
		t.Error("Expected error for empty grammar")
	}
	t.Logf("Got expected error: %v", err)
}
