package store

import (
	"context"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	// Packages
	client "github.com/mutablelogic/go-client"
	llama "github.com/mutablelogic/go-llama"
	types "github.com/mutablelogic/go-server/pkg/types"
)

///////////////////////////////////////////////////////////////////////////////
// CONSTANTS

const (
	schemeHF                = "hf"
	schemeHTTP              = "http"
	schemeHTTPS             = "https"
	hostHuggingFace         = "huggingface.co"
	defaultBranch           = "main"
	downloadParam           = "download=true"
	errInvalidHFURL         = "invalid hf:// URL: missing user"
	errInvalidHFFormat      = "invalid hf:// URL format: expected hf://user/repo/path/file"
	errInvalidHFCoURL       = "invalid huggingface.co URL format"
	errCannotParseRepo      = "could not parse repo or file path from URL"
	pullProgressMinInterval = 500 * time.Millisecond
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type Client struct {
	*client.Client
}

type ClientModel struct {
	w        io.Writer
	fn       ClientCallback
	n        uint64 // Number of bytes written
	size     uint64 // Total size from Content-Length header
	filename string // Filename from Content-Disposition header

	lastUpdate time.Time
}

var _ client.Unmarshaler = (*ClientModel)(nil)

type headUnmarshaler struct {
	headers *http.Header
}

var _ client.Unmarshaler = (*headUnmarshaler)(nil)

type ClientCallback func(filename string, bytes_received uint64, total_bytes uint64) error

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func NewClient(opts ...client.ClientOpt) (*Client, error) {
	// Create client with defaults
	defaults := []client.ClientOpt{
		client.OptEndpoint("http://localhost/"),
	}
	if c, err := client.New(append(defaults, opts...)...); err != nil {
		return nil, err
	} else {
		return &Client{Client: c}, nil
	}
}
func NewClientModel(w io.Writer, fn ClientCallback) *ClientModel {
	return &ClientModel{
		w:  w,
		fn: fn,
	}
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// GetModelSize returns the remote file size by making a HEAD request.
func (c *Client) GetModelSize(ctx context.Context, url string) (uint64, error) {
	// Parse URL to get the actual download URL
	httpURL, opts, _, err := c.parseModelUrl(url)
	if err != nil {
		return 0, err
	}

	// Perform HEAD request
	defaults := []client.RequestOpt{
		client.OptReqEndpoint(httpURL.String()),
	}
	allOpts := append(defaults, opts...)

	var headers http.Header
	headReq := client.NewRequestEx(http.MethodHead, client.ContentTypeAny)
	if err := c.DoWithContext(ctx, headReq, &headUnmarshaler{headers: &headers}, allOpts...); err != nil {
		return 0, err
	}

	// Parse Content-Length header
	if size := headers.Get(types.ContentLengthHeader); size != "" {
		if size_, err := strconv.ParseUint(size, 10, 64); err == nil {
			return size_, nil
		}
	}

	return 0, llama.ErrInvalidArgument.With("Content-Length header not found or invalid")
}

// GetDestPath returns the suggested destination path for a model URL without downloading.
func (c *Client) GetDestPath(url string) (string, error) {
	_, _, destPath, err := c.parseModelUrl(url)
	return destPath, err
}

// PullModel downloads a model from the given URL and returns the suggested destination path.
// Supports HuggingFace URLs with hf:// scheme and regular HTTP(S) URLs.
// If a callback is provided, it first makes a HEAD request and calls the callback with
// (filename, 0, total_size). If the callback returns an error, the download is aborted.
func (c *Client) PullModel(ctx context.Context, w io.Writer, url string, fn ClientCallback, additionalOpts ...client.RequestOpt) (destPath string, err error) {
	// Parse URL to get the actual download URL, options, and destination path
	httpURL, opts, destPath, err := c.parseModelUrl(url)
	if err != nil {
		return "", err
	}

	// If callback provided, do HEAD request first to get size and allow callback to abort
	if fn != nil {
		var headers http.Header
		headReq := client.NewRequestEx(http.MethodHead, client.ContentTypeAny)
		headOpts := []client.RequestOpt{client.OptReqEndpoint(httpURL.String())}
		headOpts = append(headOpts, opts...)
		headOpts = append(headOpts, additionalOpts...)

		if err := c.DoWithContext(ctx, headReq, &headUnmarshaler{headers: &headers}, headOpts...); err != nil {
			return destPath, err
		}

		// Get size and filename from headers
		var size uint64
		if sizeStr := headers.Get(types.ContentLengthHeader); sizeStr != "" {
			if size_, err := strconv.ParseUint(sizeStr, 10, 64); err == nil {
				size = size_
			}
		}

		var filename string
		if disposition := headers.Get(types.ContentDispositonHeader); disposition != "" {
			if _, params, err := mime.ParseMediaType(disposition); err == nil {
				if fn := params["filename"]; fn != "" {
					filename = fn
				}
			}
		}

		// Call callback with size info (0 bytes received so far)
		// If callback returns error, abort download
		if err := fn(filename, 0, size); err != nil {
			return destPath, err
		}
	}

	model := NewClientModel(w, fn)
	defaults := []client.RequestOpt{
		client.OptReqEndpoint(httpURL.String()),
	}
	allOpts := append(defaults, opts...)
	allOpts = append(allOpts, additionalOpts...)

	if err := c.DoWithContext(ctx, client.MethodGet, model, allOpts...); err != nil {
		// Still return the destination path even if download fails
		return destPath, err
	}

	// Return destination path and success
	return destPath, nil
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func (g *ClientModel) Unmarshal(headers http.Header, r io.Reader) error {
	// Determine the size from headers
	if size := headers.Get(types.ContentLengthHeader); size != "" {
		if size_, err := strconv.ParseUint(size, 10, 64); err == nil {
			g.size = size_
		}
	}

	// Determine the filename from headers
	if disposition := headers.Get(types.ContentDispositonHeader); disposition != "" {
		if _, params, err := mime.ParseMediaType(disposition); err == nil {
			if filename := params["filename"]; filename != "" {
				g.filename = filename
			}
		}
	}

	// Read and validate GGUF header (first 4 bytes should be "GGUF")
	magic := make([]byte, 4)
	if n, err := io.ReadFull(r, magic); err != nil {
		return err
	} else if n != 4 || string(magic) != "GGUF" {
		return llama.ErrInvalidModel.With("invalid GGUF file: expected magic header 'GGUF'")
	} else if _, err := g.Write(magic); err != nil {
		return err
	}

	// Copy the rest of the data
	if _, err := io.Copy(g, r); err != nil {
		return err
	}

	// Return success
	return nil
}

func (g *ClientModel) Write(p []byte) (int, error) {
	n, err := g.w.Write(p)
	g.n += uint64(n)
	if g.fn != nil {
		shouldNotify := g.lastUpdate.IsZero() ||
			time.Since(g.lastUpdate) >= pullProgressMinInterval ||
			(g.size > 0 && g.n >= g.size)
		if shouldNotify {
			if cbErr := g.fn(g.filename, g.n, g.size); cbErr != nil {
				return n, cbErr
			}
			g.lastUpdate = time.Now()
		}
	}
	return n, err
}

func (h *headUnmarshaler) Unmarshal(headers http.Header, r io.Reader) error {
	*h.headers = headers
	return nil
}

// parseModelUrl converts URLs into HTTP download URLs, request options, and determines the final destination path.
// Supports both hf:// scheme and regular https:// URLs
func (c *Client) parseModelUrl(urlStr string) (*url.URL, []client.RequestOpt, string, error) {
	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, nil, "", llama.ErrInvalidArgument.Withf("invalid URL: %v", err)
	}

	switch u.Scheme {
	case schemeHF:
		return parseHFScheme(u)
	case schemeHTTP, schemeHTTPS:
		return parseHTTPScheme(u)
	default:
		return nil, nil, "", llama.ErrInvalidArgument.Withf("unsupported URL scheme: %q", u.Scheme)
	}
}

// parseHFScheme handles hf://user/repo[@branch]/path/file URLs
func parseHFScheme(u *url.URL) (*url.URL, []client.RequestOpt, string, error) {
	if u.Host == "" {
		return nil, nil, "", llama.ErrInvalidArgument.With(errInvalidHFURL)
	}

	pathParts := strings.Split(strings.TrimPrefix(u.Path, "/"), "/")
	if len(pathParts) < 2 {
		return nil, nil, "", llama.ErrInvalidArgument.With(errInvalidHFFormat)
	}

	// Parse repo and branch
	repo, branch := parseRepoAndBranch(u.Host, pathParts[0])
	if len(pathParts[1:]) == 0 {
		return nil, nil, "", llama.ErrInvalidArgument.With(errInvalidHFFormat)
	}
	return buildHuggingFaceURL(repo, branch, strings.Join(pathParts[1:], "/"))
}

// parseHTTPScheme handles regular HTTP(S) URLs, including HuggingFace URLs
func parseHTTPScheme(u *url.URL) (*url.URL, []client.RequestOpt, string, error) {
	if !strings.Contains(u.Host, hostHuggingFace) {
		// Not a HuggingFace URL, return as-is with no timeout for potentially large downloads
		return u, []client.RequestOpt{client.OptNoTimeout()}, filepath.Base(u.Path), nil
	}

	pathParts := strings.Split(strings.TrimPrefix(u.Path, "/"), "/")
	if len(pathParts) < 5 {
		return nil, nil, "", llama.ErrInvalidArgument.With(errInvalidHFCoURL)
	}

	// Support both /resolve/ (API) and /blob/ (web) formats
	var branch string
	if pathParts[2] == "resolve" || pathParts[2] == "blob" {
		branch = pathParts[3]
	} else {
		return nil, nil, "", llama.ErrInvalidArgument.With(errInvalidHFCoURL)
	}

	repo := pathParts[0] + "/" + pathParts[1]
	if len(pathParts[4:]) == 0 {
		return nil, nil, "", llama.ErrInvalidArgument.With(errInvalidHFCoURL)
	}
	filePath := strings.Join(pathParts[4:], "/")

	return buildHuggingFaceURL(repo, branch, filePath)
}

// parseRepoAndBranch extracts repo and branch from user/repo[@branch] format
func parseRepoAndBranch(user, repoPart string) (repo, branch string) {
	if idx := strings.Index(repoPart, "@"); idx != -1 {
		return user + "/" + repoPart[:idx], repoPart[idx+1:]
	}
	return user + "/" + repoPart, defaultBranch
}

// buildHuggingFaceURL constructs the final HTTP download URL and destination path
func buildHuggingFaceURL(repo, branch, filePath string) (*url.URL, []client.RequestOpt, string, error) {
	if repo == "" || filePath == "" {
		return nil, nil, "", llama.ErrInvalidArgument.With(errCannotParseRepo)
	}

	urlPath, err := url.JoinPath("/", repo, "resolve", branch, filePath)
	if err != nil {
		return nil, nil, "", llama.ErrInvalidArgument.Withf("failed to construct URL path: %v", err)
	}

	// Return the parameters needed to download the file
	return &url.URL{
		Scheme:   schemeHTTPS,
		Host:     hostHuggingFace,
		Path:     urlPath,
		RawQuery: downloadParam,
	}, []client.RequestOpt{client.OptNoTimeout()}, filepath.Join(filepath.Base(repo), filepath.Base(filePath)), nil
}
