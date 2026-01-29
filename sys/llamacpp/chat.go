package llamacpp

/*
#include "chat.h"
#include "model.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// ChatMessage represents a single message in a conversation
type ChatMessage struct {
	Role    string // "system", "user", or "assistant"
	Content string // The message content
}

///////////////////////////////////////////////////////////////////////////////
// MODEL CHAT TEMPLATE

// ChatTemplate returns the chat template from model metadata
// The templateName can be empty for default, or a specific name like "tool_use"
func (m *Model) ChatTemplate(templateName string) string {
	if m.handle == nil {
		return ""
	}

	var cName *C.char
	if templateName != "" {
		cName = C.CString(templateName)
		defer C.free(unsafe.Pointer(cName))
	}

	result := C.llama_go_model_chat_template(m.handle, cName)
	if result == nil {
		return ""
	}
	return C.GoString(result)
}

// HasChatTemplate returns true if the model has a chat template
func (m *Model) HasChatTemplate() bool {
	return m.ChatTemplate("") != ""
}

///////////////////////////////////////////////////////////////////////////////
// APPLY CHAT TEMPLATE

// ApplyTemplate formats messages using the model's chat template
// If addAssistant is true, adds the assistant turn prefix at the end
func (m *Model) ApplyTemplate(messages []ChatMessage, addAssistant bool) (string, error) {
	return ApplyTemplateWithModel(m, "", messages, addAssistant)
}

// ApplyTemplateWithModel formats messages using a specific template
// If tmpl is empty, uses the model's default template
// If model is nil, tmpl must be provided
func ApplyTemplateWithModel(model *Model, tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	// Build C message array
	cMessages := make([]C.llama_go_chat_message, len(messages))
	cStrings := make([]*C.char, len(messages)*2) // role + content for each

	for i, msg := range messages {
		cStrings[i*2] = C.CString(msg.Role)
		cStrings[i*2+1] = C.CString(msg.Content)
		cMessages[i].role = cStrings[i*2]
		cMessages[i].content = cStrings[i*2+1]
	}

	// Clean up C strings when done
	defer func() {
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	// Template string
	var cTmpl *C.char
	if tmpl != "" {
		cTmpl = C.CString(tmpl)
		defer C.free(unsafe.Pointer(cTmpl))
	}

	// Model handle
	var modelHandle unsafe.Pointer
	if model != nil {
		modelHandle = model.handle
	}

	// First call to get required size
	requiredSize := C.llama_go_chat_apply_template(
		modelHandle,
		cTmpl,
		&cMessages[0],
		C.size_t(len(messages)),
		C.bool(addAssistant),
		nil,
		0,
	)

	if requiredSize < 0 {
		return "", errors.New("failed to apply chat template")
	}

	// Allocate buffer and apply template
	buf := make([]byte, requiredSize+1)
	result := C.llama_go_chat_apply_template(
		modelHandle,
		cTmpl,
		&cMessages[0],
		C.size_t(len(messages)),
		C.bool(addAssistant),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(len(buf)),
	)

	if result < 0 {
		return "", errors.New("failed to apply chat template")
	}

	return string(buf[:result]), nil
}

// ApplyBuiltinTemplate formats messages using a named built-in template
// Common templates: "llama2", "llama3", "chatml", "gemma", "phi3", etc.
func ApplyBuiltinTemplate(templateName string, messages []ChatMessage, addAssistant bool) (string, error) {
	return ApplyTemplateWithModel(nil, templateName, messages, addAssistant)
}

///////////////////////////////////////////////////////////////////////////////
// BUILT-IN TEMPLATES

// BuiltinTemplates returns the list of built-in chat template names
func BuiltinTemplates() []string {
	// Get count first
	count := C.llama_go_chat_builtin_templates(nil, 0)
	if count <= 0 {
		return nil
	}

	// Allocate array for template pointers
	templates := make([]*C.char, count)
	actual := C.llama_go_chat_builtin_templates(&templates[0], C.size_t(count))

	result := make([]string, 0, actual)
	for i := C.int32_t(0); i < actual; i++ {
		if templates[i] != nil {
			result = append(result, C.GoString(templates[i]))
		}
	}

	return result
}
