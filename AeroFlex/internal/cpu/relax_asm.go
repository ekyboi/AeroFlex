//go:build arm64 || amd64

// Package cpu provides the architecture spin-wait hint used by AeroFlux's
// lock-free barriers and turnstiles.
//
// It is a separate package because cgo-enabled packages may not contain Go
// assembly files, and the engine's Darwin build links Accelerate via cgo.
package cpu

// Relax issues the architecture's spin-wait hint.
//
//go:noescape
func Relax()
