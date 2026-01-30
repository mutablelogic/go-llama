package llamacpp

import (
	"go.opentelemetry.io/otel/trace"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type Opt func(*opt) error

// opt contains configuration for the Llama instance
type opt struct {
	tracer trace.Tracer
}

///////////////////////////////////////////////////////////////////////////////
// OPTIONS

// WithTracer sets the OpenTelemetry tracer for distributed tracing of
// model loading, unloading, and other operations.
func WithTracer(tracer trace.Tracer) Opt {
	return func(o *opt) error {
		o.tracer = tracer
		return nil
	}
}
