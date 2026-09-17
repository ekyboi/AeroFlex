//go:build darwin && cgo

package aeroflux

import (
	"fmt"
	"unsafe"
)

// EnableSparseSolver assembles the 7-point pressure Poisson matrix in arena
// memory and Cholesky-factorizes it through Accelerate.
//
// The factorization is computed once and reused for every timestep, which
// is the entire reason to prefer it over the iterative sweeps: the Poisson
// operator is structurally constant for a fixed grid, so the expensive part
// happens exactly once and each step pays only two triangular solves. The
// iterative path re-does all its work every step.
//
// Boundary handling: solid walls impose homogeneous Neumann, so a wall
// neighbor simply drops out of the stencil and the diagonal is reduced
// accordingly. That leaves the matrix singular up to a constant (pressure
// is defined only up to an offset), which Cholesky cannot factor. The
// standard fix, used here, is to pin one cell by adding a small positive
// value to a single diagonal entry, selecting the zero-mean solution
// without perturbing the gradient that actually gets used.
func (e *Engine) EnableSparseSolver() error {
	if e.sparse != nil {
		return nil
	}
	m := e.Mesh
	n := m.CellCount()

	// The lower triangle of a 7-point Laplacian holds, per column, the
	// diagonal plus at most three forward-neighbor couplings (+x, +y, +z).
	maxNNZ := n * 4

	// Lay the CSC arrays out in a dedicated arena so they outlive the
	// factorization -- Accelerate borrows these pointers rather than
	// copying them, so they must not be scratch memory the solver reuses.
	csBytes := (n + 1) * int(unsafe.Sizeof(int64(0)))
	riBytes := maxNNZ * int(unsafe.Sizeof(int32(0)))
	vaBytes := maxNNZ * int(unsafe.Sizeof(float64(0)))

	csOff := 0
	riOff := align8(csOff + csBytes)
	vaOff := align8(riOff + riBytes)
	rhsOff := align8(vaOff + vaBytes)
	total := rhsOff + n*int(unsafe.Sizeof(float64(0)))

	ma, err := NewArena(total)
	if err != nil {
		return fmt.Errorf("aeroflux: matrix arena: %w", err)
	}

	cs := unsafe.Slice((*int64)(unsafe.Pointer(ma.Pointer(csOff))), n+1)
	ri := unsafe.Slice((*int32)(unsafe.Pointer(ma.Pointer(riOff))), maxNNZ)
	va := unsafe.Slice((*float64)(unsafe.Pointer(ma.Pointer(vaOff))), maxNNZ)

	h2 := m.H * m.H
	k := 0
	for kk := 0; kk < m.NZ; kk++ {
		for jj := 0; jj < m.NY; jj++ {
			for ii := 0; ii < m.NX; ii++ {
				col := m.CellIndex(ii, jj, kk)
				cs[col] = int64(k)

				// Diagonal: one unit per non-wall neighbor. A wall
				// neighbor is absent from the stencil under Neumann, so
				// it lowers the diagonal instead of contributing a row.
				diag := 0.0
				if ii > 0 {
					diag++
				}
				if ii < m.NX-1 {
					diag++
				}
				if jj > 0 {
					diag++
				}
				if jj < m.NY-1 {
					diag++
				}
				if kk > 0 {
					diag++
				}
				if kk < m.NZ-1 {
					diag++
				}

				// Pin the first cell to fix the additive constant and make
				// the system positive-definite rather than semi-definite.
				if col == 0 {
					diag += 1.0
				}

				ri[k] = int32(col)
				va[k] = diag / h2
				k++

				// Only forward couplings: the lower triangle in CSC order
				// requires ascending row indices within each column, and
				// +x < +y < +z holds by construction of CellIndex.
				if ii < m.NX-1 {
					ri[k] = int32(m.CellIndex(ii+1, jj, kk))
					va[k] = -1.0 / h2
					k++
				}
				if jj < m.NY-1 {
					ri[k] = int32(m.CellIndex(ii, jj+1, kk))
					va[k] = -1.0 / h2
					k++
				}
				if kk < m.NZ-1 {
					ri[k] = int32(m.CellIndex(ii, jj, kk+1))
					va[k] = -1.0 / h2
					k++
				}
			}
		}
	}
	cs[n] = int64(k)

	s, err := NewPoissonSolver(ma.Pointer(csOff), ma.Pointer(riOff), ma.Pointer(vaOff), n, k)
	if err != nil {
		_ = ma.Close()
		return err
	}
	e.sparse = s
	e.matArena = ma
	e.rhsOff = rhsOff
	e.Cfg.UseSparseSolver = true
	return nil
}

// solvePressureDirect solves the Poisson system with the prefactorized
// Cholesky decomposition.
//
// Div holds the RHS on the padded grid but Accelerate needs a dense,
// compact vector of interior cells, so the RHS is gathered into a
// contiguous arena scratch buffer, solved in place, and scattered back.
// The gather/scatter is arena-to-arena and allocates nothing.
func (e *Engine) solvePressureDirect() error {
	m := e.Mesh
	n := m.CellCount()
	g := m.Ghost

	rhs := unsafe.Slice((*float64)(unsafe.Pointer(e.matArena.Pointer(e.rhsOff))), n)
	dv := m.Data(&m.Div)

	for kk := 0; kk < m.NZ; kk++ {
		for jj := 0; jj < m.NY; jj++ {
			for ii := 0; ii < m.NX; ii++ {
				rhs[m.CellIndex(ii, jj, kk)] = -dv[m.Div.Index(ii+g, jj+g, kk+g)]
			}
		}
	}

	if err := e.sparse.SolveInPlace(e.matArena.Pointer(e.rhsOff)); err != nil {
		return err
	}

	p := m.Data(&m.P)
	for kk := 0; kk < m.NZ; kk++ {
		for jj := 0; jj < m.NY; jj++ {
			for ii := 0; ii < m.NX; ii++ {
				p[m.P.Index(ii+g, jj+g, kk+g)] = rhs[m.CellIndex(ii, jj, kk)]
			}
		}
	}
	e.applyPressureBC()
	return nil
}

func align8(n int) int {
	if r := n % 8; r != 0 {
		return n + (8 - r)
	}
	return n
}
