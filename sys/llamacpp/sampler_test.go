package llamacpp_test

import (
	"testing"

	"github.com/mutablelogic/go-llama/sys/llamacpp"
)

const testModelSampler = "../../testdata/stories260K.gguf"

func TestSamplerNew(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelSampler, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Create sampler with default params
	sampler, err := llamacpp.NewSampler(model, llamacpp.DefaultSamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	if sampler == nil {
		t.Fatal("expected non-nil sampler")
	}
	defer sampler.Close()
}

func TestSamplerGreedy(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelSampler, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Create greedy sampler
	sampler, err := llamacpp.NewSampler(model, llamacpp.GreedySamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	if sampler == nil {
		t.Fatal("expected non-nil sampler")
	}
	defer sampler.Close()
}

func TestSamplerCustomParams(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelSampler, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	params := llamacpp.SamplerParams{
		Seed:             42,
		Temperature:      0.5,
		TopK:             20,
		TopP:             0.9,
		MinP:             0.1,
		RepeatPenalty:    1.2,
		RepeatLastN:      32,
		FrequencyPenalty: 0.1,
		PresencePenalty:  0.1,
	}

	sampler, err := llamacpp.NewSampler(model, params)
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	if sampler == nil {
		t.Fatal("expected non-nil sampler")
	}
	defer sampler.Close()
}

func TestSamplerReset(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelSampler, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	sampler, err := llamacpp.NewSampler(model, llamacpp.DefaultSamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	defer sampler.Close()

	// Reset should not panic
	sampler.Reset()
}

func TestSamplerChainEmpty(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Create an empty chain
	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()
}

func TestSamplerChainGreedy(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Create a chain with just greedy sampler
	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	chain.AddGreedy()
}

func TestSamplerChainTemp(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	chain.AddTemp(0.8)
	chain.AddDist(42)
}

func TestSamplerChainTopK(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	chain.AddTopK(40)
	chain.AddDist(42)
}

func TestSamplerChainTopP(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	chain.AddTopP(0.95, 1)
	chain.AddDist(42)
}

func TestSamplerChainMinP(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	chain.AddMinP(0.05, 1)
	chain.AddDist(42)
}

func TestSamplerChainPenalties(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	chain.AddPenalties(64, 1.1, 0.0, 0.0)
	chain.AddDist(42)
}

func TestSamplerChainFull(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	// Create a full chain similar to default params
	chain := llamacpp.NewSamplerChain()
	if chain == nil {
		t.Fatal("expected non-nil chain")
	}
	defer chain.Close()

	// Add samplers in recommended order
	chain.AddPenalties(64, 1.1, 0.0, 0.0) // repeat penalty
	chain.AddTopK(40)
	chain.AddTopP(0.95, 1)
	chain.AddMinP(0.05, 1)
	chain.AddTemp(0.8)
	chain.AddDist(0) // random seed

	// Verify chain length
	if chain.Length() != 6 {
		t.Errorf("expected chain length 6, got %d", chain.Length())
	}
}

func TestSamplerParamsDefaults(t *testing.T) {
	params := llamacpp.DefaultSamplerParams()

	if params.Temperature != 0.8 {
		t.Errorf("expected temperature 0.8, got %f", params.Temperature)
	}
	if params.TopK != 40 {
		t.Errorf("expected top_k 40, got %d", params.TopK)
	}
	if params.TopP != 0.95 {
		t.Errorf("expected top_p 0.95, got %f", params.TopP)
	}
	if params.MinP != 0.05 {
		t.Errorf("expected min_p 0.05, got %f", params.MinP)
	}
	if params.RepeatPenalty != 1.1 {
		t.Errorf("expected repeat_penalty 1.1, got %f", params.RepeatPenalty)
	}
	if params.RepeatLastN != 64 {
		t.Errorf("expected repeat_last_n 64, got %d", params.RepeatLastN)
	}
}

func TestSamplerParamsGreedy(t *testing.T) {
	params := llamacpp.GreedySamplerParams()

	if params.Temperature != 0.0 {
		t.Errorf("expected temperature 0.0, got %f", params.Temperature)
	}
}

func TestSamplerAccept(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelSampler, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	sampler, err := llamacpp.NewSampler(model, llamacpp.DefaultSamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	defer sampler.Close()

	// Accept should not panic (token 1 is usually valid)
	sampler.Accept(1)
}

func TestSamplerChainLength(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	modelParams := llamacpp.DefaultModelParams()
	model, err := llamacpp.LoadModel(testModelSampler, modelParams)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	sampler, err := llamacpp.NewSampler(model, llamacpp.DefaultSamplerParams())
	if err != nil {
		t.Fatalf("failed to create sampler: %v", err)
	}
	defer sampler.Close()

	// Default sampler should have multiple samplers in chain
	length := sampler.ChainLength()
	if length <= 0 {
		t.Errorf("expected positive chain length, got %d", length)
	}
	t.Logf("Default sampler chain length: %d", length)
}

func TestXTCSampler(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultXTCParams()
	params.Probability = 0.1
	params.Threshold = 0.1

	sampler, err := llamacpp.NewXTCSampler(params)
	if err != nil {
		t.Fatalf("failed to create XTC sampler: %v", err)
	}
	defer sampler.Close()

	t.Log("XTC sampler created successfully")
}

func TestMirostatV2Sampler(t *testing.T) {
	llamacpp.Init()
	defer llamacpp.Cleanup()

	params := llamacpp.DefaultMirostatV2Params()
	params.Tau = 5.0
	params.Eta = 0.1

	sampler, err := llamacpp.NewMirostatV2Sampler(params)
	if err != nil {
		t.Fatalf("failed to create Mirostat v2 sampler: %v", err)
	}
	defer sampler.Close()

	t.Log("Mirostat v2 sampler created successfully")
}
