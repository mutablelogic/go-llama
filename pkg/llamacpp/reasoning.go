package llamacpp

import (
	"regexp"
	"strings"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// ReasoningResult contains the parsed output from a reasoning model
type ReasoningResult struct {
	// Thinking contains the model's reasoning/thinking process
	// This is typically hidden from end users
	Thinking string

	// Content contains the final response after reasoning
	// This is what should be shown to users
	Content string

	// HasThinking indicates whether thinking tags were found
	HasThinking bool
}

///////////////////////////////////////////////////////////////////////////////
// COMMON PATTERNS

// Common thinking tag patterns used by various models
var (
	// DeepSeek R1 style: <think>...</think>
	thinkPattern = regexp.MustCompile(`(?s)<think>(.*?)</think>`)

	// Alternative patterns
	reasoningPattern  = regexp.MustCompile(`(?s)<reasoning>(.*?)</reasoning>`)
	scratchpadPattern = regexp.MustCompile(`(?s)<scratchpad>(.*?)</scratchpad>`)
	internalPattern   = regexp.MustCompile(`(?s)<internal>(.*?)</internal>`)

	// Claude-style artifact pattern (not exactly thinking, but similar)
	thoughtPattern = regexp.MustCompile(`(?s)<thought>(.*?)</thought>`)
)

///////////////////////////////////////////////////////////////////////////////
// EXTRACTION FUNCTIONS

// ParseReasoning extracts thinking/reasoning blocks from model output
// It supports multiple common formats:
//   - <think>...</think> (DeepSeek R1)
//   - <reasoning>...</reasoning>
//   - <scratchpad>...</scratchpad>
//   - <thought>...</thought>
//   - <internal>...</internal>
//
// Returns a ReasoningResult with separated thinking and content
func ParseReasoning(text string) ReasoningResult {
	result := ReasoningResult{
		Content:     text,
		HasThinking: false,
	}

	// Try each pattern in order of commonality
	patterns := []*regexp.Regexp{
		thinkPattern,
		reasoningPattern,
		scratchpadPattern,
		thoughtPattern,
		internalPattern,
	}

	var allThinking []string
	remaining := text

	for _, pattern := range patterns {
		matches := pattern.FindAllStringSubmatch(remaining, -1)
		for _, match := range matches {
			if len(match) >= 2 {
				allThinking = append(allThinking, strings.TrimSpace(match[1]))
				result.HasThinking = true
			}
		}
		// Remove matched patterns from remaining text
		remaining = pattern.ReplaceAllString(remaining, "")
	}

	if result.HasThinking {
		result.Thinking = strings.Join(allThinking, "\n\n")
		result.Content = strings.TrimSpace(remaining)
	}

	return result
}

// ParseReasoningWithTag extracts content from a specific tag pattern
// tag should be the tag name without brackets (e.g., "think" not "<think>")
func ParseReasoningWithTag(text, tag string) ReasoningResult {
	result := ReasoningResult{
		Content:     text,
		HasThinking: false,
	}

	// Build pattern for the specific tag
	pattern := regexp.MustCompile(`(?s)<` + regexp.QuoteMeta(tag) + `>(.*?)</` + regexp.QuoteMeta(tag) + `>`)

	var allThinking []string
	matches := pattern.FindAllStringSubmatch(text, -1)
	for _, match := range matches {
		if len(match) >= 2 {
			allThinking = append(allThinking, strings.TrimSpace(match[1]))
			result.HasThinking = true
		}
	}

	if result.HasThinking {
		result.Thinking = strings.Join(allThinking, "\n\n")
		result.Content = strings.TrimSpace(pattern.ReplaceAllString(text, ""))
	}

	return result
}

// ExtractThinkingBlocks returns all thinking blocks found in the text
// without modifying the original text
func ExtractThinkingBlocks(text string) []string {
	var blocks []string

	patterns := []*regexp.Regexp{
		thinkPattern,
		reasoningPattern,
		scratchpadPattern,
		thoughtPattern,
		internalPattern,
	}

	for _, pattern := range patterns {
		matches := pattern.FindAllStringSubmatch(text, -1)
		for _, match := range matches {
			if len(match) >= 2 {
				blocks = append(blocks, strings.TrimSpace(match[1]))
			}
		}
	}

	return blocks
}

// StripThinking removes all thinking/reasoning tags from text
// leaving only the final response content
func StripThinking(text string) string {
	result := text

	patterns := []*regexp.Regexp{
		thinkPattern,
		reasoningPattern,
		scratchpadPattern,
		thoughtPattern,
		internalPattern,
	}

	for _, pattern := range patterns {
		result = pattern.ReplaceAllString(result, "")
	}

	return strings.TrimSpace(result)
}

// HasThinkingTags checks if the text contains any thinking/reasoning tags
func HasThinkingTags(text string) bool {
	patterns := []*regexp.Regexp{
		thinkPattern,
		reasoningPattern,
		scratchpadPattern,
		thoughtPattern,
		internalPattern,
	}

	for _, pattern := range patterns {
		if pattern.MatchString(text) {
			return true
		}
	}

	return false
}

// IsThinkingTemplate checks if a chat template string suggests it's a reasoning template
// by looking for common thinking-related patterns
func IsThinkingTemplate(template string) bool {
	if template == "" {
		return false
	}

	// Check for common thinking model indicators in templates
	thinkingIndicators := []string{
		"<think>",
		"</think>",
		"<reasoning>",
		"</reasoning>",
		"<scratchpad>",
		"thinking",
		"reason",
	}

	templateLower := strings.ToLower(template)
	for _, indicator := range thinkingIndicators {
		if strings.Contains(templateLower, indicator) {
			return true
		}
	}

	return false
}

// IsThinkingModelName checks if a model name/architecture suggests it's a reasoning model
func IsThinkingModelName(modelInfo string) bool {
	combined := strings.ToLower(modelInfo)

	reasoningModels := []string{
		"deepseek",
		"r1",
		"reasoning",
		"think",
	}

	for _, model := range reasoningModels {
		if strings.Contains(combined, model) {
			return true
		}
	}

	return false
}
