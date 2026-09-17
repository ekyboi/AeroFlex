//go:build !darwin || !cgo

package aeroflux

import "errors"

// PoissonSolver is unavailable without Accelerate. The type exists so the
// Engine struct has one shape across platforms.
type PoissonSolver struct{}

// Close is a no-op on platforms without Accelerate.
func (s *PoissonSolver) Close() {}

// EnableSparseSolver reports that no direct solver is available here. The
// iterative Red-Black path is fully functional and is what the engine uses
// on these platforms.
func (e *Engine) EnableSparseSolver() error {
	return errors.New("aeroflux: sparse direct solver requires darwin with cgo (Accelerate); the Red-Black iterative solver is used instead")
}

// solvePressureDirect is never reached: UseSparseSolver can only be set by
// a successful EnableSparseSolver, which always fails here.
func (e *Engine) solvePressureDirect() error {
	return errors.New("aeroflux: no direct solver on this platform")
}
