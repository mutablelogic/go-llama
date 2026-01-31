package schema

//////////////////////////////////////////////////////////////////////////////
// GLOBALS

const (
	CompletionStreamDeltaType = "completion.delta"
	CompletionStreamDoneType  = "completion.done"
	CompletionStreamErrorType = "completion.error"

	ModelPullProgressType = "model.pull.progress"
	ModelPullCompleteType = "model.pull.complete"
	ModelPullErrorType    = "model.pull.error"
)

//////////////////////////////////////////////////////////////////////////////
// TYPES

// ModelPullProgress represents progress information during model download
type ModelPullProgress struct {
	Filename      string  `json:"model"`
	BytesReceived uint64  `json:"bytes_received"`
	TotalBytes    uint64  `json:"total_bytes,omitempty"`
	Percentage    float64 `json:"percent,omitempty"`
}
