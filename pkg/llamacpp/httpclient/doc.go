// Package httpclient provides a typed Go client for consuming the go-llama
// REST API.
//
// Create a client with:
//
//	client, err := httpclient.New("http://localhost:8080/api/gollama")
//	if err != nil {
//	    panic(err)
//	}
//
// Then use the client to manage models and perform inference:
//
//	// List all models
//	models, err := client.ListModels(ctx)
//
//	// Get a specific model
//	model, err := client.GetModel(ctx, "llama-7b")
//
//	// Load a model into memory
//	model, err := client.LoadModel(ctx, "llama-7b")
//
//	// Unload a model from memory
//	err := client.UnloadModel(ctx, "llama-7b")
//
//	// Generate text completion
//	result, err := client.Complete(ctx, "llama-7b", "Once upon a time",
//	    httpclient.WithMaxTokens(100),
//	    httpclient.WithTemperature(0.7))
//
//	// Stream completion tokens
//	result, err := client.Complete(ctx, "llama-7b", "Once upon a time",
//	    httpclient.WithChunkCallback(func(chunk *schema.CompletionChunk) error {
//	        fmt.Print(chunk.Text)
//	        return nil
//	    }))
//
//	// Generate embeddings
//	result, err := client.Embed(ctx, "embedding-model", []string{"Hello", "World"})
//
//	// Tokenize text
//	tokens, err := client.Tokenize(ctx, "llama-7b", "Hello, world!")
//
//	// Detokenize tokens
//	text, err := client.Detokenize(ctx, "llama-7b", tokens.Tokens)
package httpclient
