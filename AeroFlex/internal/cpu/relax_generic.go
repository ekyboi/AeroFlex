//go:build !arm64 && !amd64

package cpu

import "runtime"

// Relax falls back to a scheduler yield on architectures without a
// dedicated spin hint. Correct everywhere, just less efficient.
func Relax() { runtime.Gosched() }
