package schema

import "strings"

const spanPrefix = "llamacpp"

// SpanName returns a span name with "llamacpp" prefix joined by periods.
// SpanName("ListModels") returns "llamacpp.ListModels"
// SpanName("Model", "Load") returns "llamacpp.Model.Load"
func SpanName(args ...string) string {
	parts := make([]string, 0, len(args)+1)
	parts = append(parts, spanPrefix)
	parts = append(parts, args...)
	return strings.Join(parts, ".")
}
