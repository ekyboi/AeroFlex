//go:build darwin && cgo

package aeroflux

import (
	"math"
	"testing"
	"unsafe"
)

// buildLaplacian1D writes a symmetric tridiagonal 1D Poisson operator
// (-1, 2, -1) into arena memory in CSC form with only the lower triangle
// stored, which is what NewPoissonSolver expects.
//
// This is the same structural pattern the 3D 7-point stencil uses, just
// small enough to verify against an exact analytic solution.
func buildLaplacian1D(a *Arena, n int) (colStarts, rowIdx, vals uintptr, nnz int) {
	// Lower triangle of a tridiagonal matrix: diagonal + one subdiagonal
	// per column except the last.
	nnz = n + (n - 1)

	csOff := 0
	riOff := csOff + (n+1)*int(unsafe.Sizeof(int64(0)))
	vaOff := riOff + nnz*int(unsafe.Sizeof(int32(0)))
	// Keep the float64 array 8-byte aligned.
	if vaOff%8 != 0 {
		vaOff += 8 - vaOff%8
	}

	cs := unsafe.Slice((*int64)(unsafe.Pointer(a.Pointer(csOff))), n+1)
	ri := unsafe.Slice((*int32)(unsafe.Pointer(a.Pointer(riOff))), nnz)
	va := unsafe.Slice((*float64)(unsafe.Pointer(a.Pointer(vaOff))), nnz)

	k := 0
	for j := 0; j < n; j++ {
		cs[j] = int64(k)
		ri[k] = int32(j)
		va[k] = 2.0
		k++
		if j+1 < n {
			ri[k] = int32(j + 1)
			va[k] = -1.0
			k++
		}
	}
	cs[n] = int64(k)

	return a.Pointer(csOff), a.Pointer(riOff), a.Pointer(vaOff), nnz
}

// TestPoissonSolveAgainstKnownSolution factorizes a 1D Laplacian and solves
// it against a right-hand side with an exact known answer. If the CSC
// layout, triangle attribute, or element sizes were wrong, this produces
// garbage rather than merely being slow -- which is exactly why it is
// tested against analytic truth rather than self-consistency.
func TestPoissonSolveAgainstKnownSolution(t *testing.T) {
	a, err := NewArena(4 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const n = 255
	cs, ri, va, nnz := buildLaplacian1D(a, n)

	s, err := NewPoissonSolver(cs, ri, va, n, nnz)
	if err != nil {
		t.Fatalf("NewPoissonSolver: %v", err)
	}
	defer s.Close()

	// Choose x_true, compute b = A x_true, then verify the solver recovers
	// x_true from b.
	rhsOff := 3 << 20
	if rhsOff%8 != 0 {
		t.Fatal("rhs offset must be 8-byte aligned")
	}
	b := a.Float64s(rhsOff, n)

	xTrue := make([]float64, n) // test-only heap use, outside the hot path
	for i := range xTrue {
		xTrue[i] = math.Sin(float64(i+1) * math.Pi / float64(n+1))
	}
	for i := 0; i < n; i++ {
		v := 2 * xTrue[i]
		if i > 0 {
			v -= xTrue[i-1]
		}
		if i+1 < n {
			v -= xTrue[i+1]
		}
		b[i] = v
	}

	if err := s.SolveInPlace(a.Pointer(rhsOff)); err != nil {
		t.Fatalf("SolveInPlace: %v", err)
	}

	for i := 0; i < n; i++ {
		if math.Abs(b[i]-xTrue[i]) > 1e-9 {
			t.Fatalf("solution mismatch at %d: got %v want %v", i, b[i], xTrue[i])
		}
	}
}

// TestPoissonSolveZeroAllocs confirms the per-timestep solve path does not
// touch the Go heap.
func TestPoissonSolveZeroAllocs(t *testing.T) {
	a, err := NewArena(4 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const n = 127
	cs, ri, va, nnz := buildLaplacian1D(a, n)

	s, err := NewPoissonSolver(cs, ri, va, n, nnz)
	if err != nil {
		t.Fatalf("NewPoissonSolver: %v", err)
	}
	defer s.Close()

	rhsOff := 3 << 20
	b := a.Float64s(rhsOff, n)
	for i := range b {
		b[i] = 1.0
	}

	rhsPtr := a.Pointer(rhsOff)
	got := testing.AllocsPerRun(50, func() {
		_ = s.SolveInPlace(rhsPtr)
	})
	if got != 0 {
		t.Errorf("SolveInPlace allocated %v times per run, want 0", got)
	}
}
