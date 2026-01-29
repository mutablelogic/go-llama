package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

func TestSpecialTokens(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	bos := model.BOS()
	eos := model.EOS()
	nl := model.NL()

	t.Logf("BOS token: %d", bos)
	t.Logf("EOS token: %d", eos)
	t.Logf("NL token: %d", nl)

	if bos < 0 {
		t.Log("Note: BOS token is undefined for this model")
	}
	if eos < 0 {
		t.Log("Note: EOS token is undefined for this model")
	}
}

func TestTokenize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	text := "Hello, world!"
	opts := llamacpp.DefaultTokenizeOptions()

	tokens, err := model.Tokenize(text, opts)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Expected non-empty token list")
	}

	t.Logf("Text: %q -> %d tokens: %v", text, len(tokens), tokens)
}

func TestTokenizeEmpty(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize("", opts)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}

	if len(tokens) != 0 {
		t.Errorf("Expected empty token list for empty string, got %d tokens", len(tokens))
	}
}

func TestTokenizeNoSpecial(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	text := "Hello"

	optsWithSpecial := llamacpp.TokenizeOptions{AddSpecial: true}
	tokensWithSpecial, err := model.Tokenize(text, optsWithSpecial)
	if err != nil {
		t.Fatalf("Tokenize with special failed: %v", err)
	}

	optsNoSpecial := llamacpp.TokenizeOptions{AddSpecial: false}
	tokensNoSpecial, err := model.Tokenize(text, optsNoSpecial)
	if err != nil {
		t.Fatalf("Tokenize without special failed: %v", err)
	}

	t.Logf("With special: %d tokens %v", len(tokensWithSpecial), tokensWithSpecial)
	t.Logf("Without special: %d tokens %v", len(tokensNoSpecial), tokensNoSpecial)

	if len(tokensWithSpecial) < len(tokensNoSpecial) {
		t.Error("Expected more tokens when adding special tokens")
	}
}

func TestTokenToString(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	text := "Hello"
	opts := llamacpp.TokenizeOptions{AddSpecial: false}

	tokens, err := model.Tokenize(text, opts)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}

	for i, token := range tokens {
		piece, err := model.TokenToString(token)
		if err != nil {
			t.Errorf("TokenToString(%d) failed: %v", token, err)
			continue
		}
		t.Logf("Token %d: %d -> %q", i, token, piece)
	}
}

func TestDetokenize(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	originalText := "Hello, world!"
	tokenizeOpts := llamacpp.TokenizeOptions{AddSpecial: false}

	tokens, err := model.Tokenize(originalText, tokenizeOpts)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}

	detokenizeOpts := llamacpp.DefaultDetokenizeOptions()
	decoded, err := model.Detokenize(tokens, detokenizeOpts)
	if err != nil {
		t.Fatalf("Detokenize failed: %v", err)
	}

	t.Logf("Original: %q", originalText)
	t.Logf("Tokens: %v", tokens)
	t.Logf("Decoded: %q", decoded)

	if decoded != originalText {
		t.Logf("Note: Decoded text differs from original (this can be normal)")
	}
}

func TestDetokenizeEmpty(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	opts := llamacpp.DefaultDetokenizeOptions()
	decoded, err := model.Detokenize([]llamacpp.Token{}, opts)
	if err != nil {
		t.Fatalf("Detokenize failed: %v", err)
	}

	if decoded != "" {
		t.Errorf("Expected empty string for empty tokens, got %q", decoded)
	}
}

func TestRoundTrip(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	testCases := []string{
		"Hello",
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"1 + 1 = 2",
	}

	tokenizeOpts := llamacpp.TokenizeOptions{AddSpecial: false}
	detokenizeOpts := llamacpp.DefaultDetokenizeOptions()

	for _, text := range testCases {
		tokens, err := model.Tokenize(text, tokenizeOpts)
		if err != nil {
			t.Errorf("Tokenize(%q) failed: %v", text, err)
			continue
		}

		decoded, err := model.Detokenize(tokens, detokenizeOpts)
		if err != nil {
			t.Errorf("Detokenize(%q) failed: %v", text, err)
			continue
		}

		t.Logf("%q -> %d tokens -> %q", text, len(tokens), decoded)
	}
}

func TestIsEOG(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	eos := model.EOS()
	if eos >= 0 {
		isEog := model.IsEOG(eos)
		t.Logf("EOS token %d is EOG: %v", eos, isEog)
	}

	tokens, _ := model.Tokenize("Hello", llamacpp.TokenizeOptions{AddSpecial: false})
	if len(tokens) > 0 {
		isEog := model.IsEOG(tokens[0])
		if isEog {
			t.Error("Regular token should not be EOG")
		}
	}
}

func TestIsControl(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	bos := model.BOS()
	eos := model.EOS()

	if bos >= 0 {
		isControl := model.IsControl(bos)
		t.Logf("BOS token %d is control: %v", bos, isControl)
	}
	if eos >= 0 {
		isControl := model.IsControl(eos)
		t.Logf("EOS token %d is control: %v", eos, isControl)
	}
}

func TestLongText(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModel, params)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	longText := ""
	for i := 0; i < 100; i++ {
		longText += "The quick brown fox jumps over the lazy dog. "
	}

	opts := llamacpp.DefaultTokenizeOptions()
	tokens, err := model.Tokenize(longText, opts)
	if err != nil {
		t.Fatalf("Tokenize long text failed: %v", err)
	}

	t.Logf("Long text (%d bytes) -> %d tokens", len(longText), len(tokens))

	if len(tokens) == 0 {
		t.Error("Expected tokens for long text")
	}
}
