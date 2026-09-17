//go:build darwin && cgo

package aeroflux

/*
#cgo LDFLAGS: -framework Accelerate
#cgo CFLAGS: -O3

#include <Accelerate/Accelerate.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// vForce shims.
//
// vForce's signatures take (double *y, const double *x, const int *n). The
// count is passed by POINTER, not by value -- a Fortran-derived convention
// that is easy to get wrong from cgo and silently reads garbage as the
// length. These shims take the count by value and materialize the pointer
// on the C stack, so the Go side cannot make that mistake, and they take
// addresses as uintptr-compatible integers so no Go pointer is ever passed
// across the cgo boundary (cgo pointer-passing rules would otherwise reject
// or pin them).
// ---------------------------------------------------------------------------

static void af_vvsin(uintptr_t y, uintptr_t x, int n) {
    vvsin((double *)y, (const double *)x, &n);
}

static void af_vvcos(uintptr_t y, uintptr_t x, int n) {
    vvcos((double *)y, (const double *)x, &n);
}

static void af_vvexp(uintptr_t y, uintptr_t x, int n) {
    vvexp((double *)y, (const double *)x, &n);
}

// vDSP fused multiply-add over a strided vector: y[i] = x[i]*a + b.
// Used on the advection path where a phase term is scaled and biased in one
// pass rather than two.
static void af_vsmsa(uintptr_t x, double a, double b, uintptr_t y, long n) {
    vDSP_vsmsaD((const double *)x, 1, &a, &b, (double *)y, 1, (vDSP_Length)n);
}

// ---------------------------------------------------------------------------
// Sparse solver: Red-Black Gauss-Seidel Poisson operator.
//
// The pressure Poisson matrix for a 3D 7-point Laplacian is symmetric and
// structurally constant for a fixed grid, so the factorization is computed
// ONCE at setup and reused every timestep. That is the whole reason to go
// through SparseFactor rather than hand-rolling the sweeps: the per-step
// cost collapses to triangular solves.
//
// Ownership note: SparseMatrix_Double borrows the caller's index and value
// arrays; it does not copy them. They must therefore live in the arena and
// outlive the factorization, which is exactly what AF_PoissonSetup assumes.
// ---------------------------------------------------------------------------

typedef struct {
    SparseOpaqueFactorization_Double factorization;
    SparseMatrix_Double              matrix;
    int                              valid;
} AF_Poisson;

// AF_PoissonFactor builds a symmetric sparse matrix in CSC form from arena-
// resident arrays and Cholesky-factorizes it.
//
// rowIndices/columnStarts/values all point into the arena. n is the column
// count (== number of interior pressure cells). Returns 0 on success, or
// the SparseStatus on failure.
static int AF_PoissonFactor(AF_Poisson *p,
                            uintptr_t columnStarts,
                            uintptr_t rowIndices,
                            uintptr_t values,
                            int n,
                            int nnz) {
    SparseMatrixStructure structure;
    structure.rowCount     = n;
    structure.columnCount  = n;
    structure.columnStarts = (long *)columnStarts;
    structure.rowIndices   = (int *)rowIndices;
    structure.attributes   = (SparseAttributes_t){
        .transpose      = false,
        // Only the lower triangle is stored; the operator is symmetric.
        .triangle       = SparseLowerTriangle,
        .kind           = SparseSymmetric,
        ._reserved      = 0,
        ._allocatedBySparse = false,
    };
    structure.blockSize = 1;

    p->matrix.structure = structure;
    p->matrix.data      = (double *)values;

    p->factorization = SparseFactor(SparseFactorizationCholesky, p->matrix);

    SparseStatus_t st = p->factorization.status;
    if (st != SparseStatusOK) {
        p->valid = 0;
        return (int)st;
    }
    p->valid = 1;
    (void)nnz;
    return 0;
}

// AF_PoissonSolve solves A x = b in place over arena memory. b is
// overwritten with the solution, so no temporary buffer is allocated
// anywhere in the call -- the zero-allocation requirement holds through
// the solver.
static int AF_PoissonSolve(AF_Poisson *p, uintptr_t b, int n) {
    if (!p->valid) return -1;

    DenseVector_Double rhs;
    rhs.count = n;
    rhs.data  = (double *)b;

    SparseSolve(p->factorization, rhs);
    return 0;
}

static void AF_PoissonFree(AF_Poisson *p) {
    if (p->valid) {
        SparseCleanup(p->factorization);
        p->valid = 0;
    }
}

// ---------------------------------------------------------------------------
// Mach thread affinity.
//
// thread_policy_set is a Mach trap, not a BSD syscall, so it is unreachable
// via syscall.Syscall and must be called through cgo. The affinity tag is a
// HINT: threads sharing a nonzero tag are preferentially scheduled onto one
// L2 cluster. It cannot pin to a specific core and cannot guarantee P-core
// placement -- no userspace API on macOS can.
// ---------------------------------------------------------------------------

static int AF_SetClusterAffinity(int32_t tag) {
    thread_affinity_policy_data_t policy = { .affinity_tag = tag };
    kern_return_t kr = thread_policy_set(
        mach_thread_self(),
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT);
    return (int)kr;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// VecSin computes dst[i] = sin(src[i]) for n elements, dispatching to
// Accelerate's vForce. Both addresses must be arena-resident and 8-byte
// aligned.
//
// Addresses cross the cgo boundary as uintptr, never as Go pointers, so
// cgo's pointer-passing checks are satisfied by construction and nothing
// is pinned or copied. This is only sound because the memory is off-heap:
// the GC can neither move nor collect it.
func VecSin(dst, src uintptr, n int) {
	if n <= 0 {
		return
	}
	C.af_vvsin(C.uintptr_t(dst), C.uintptr_t(src), C.int(n))
}

// VecCos computes dst[i] = cos(src[i]) for n elements.
func VecCos(dst, src uintptr, n int) {
	if n <= 0 {
		return
	}
	C.af_vvcos(C.uintptr_t(dst), C.uintptr_t(src), C.int(n))
}

// VecExp computes dst[i] = exp(src[i]) for n elements.
func VecExp(dst, src uintptr, n int) {
	if n <= 0 {
		return
	}
	C.af_vvexp(C.uintptr_t(dst), C.uintptr_t(src), C.int(n))
}

// VecScaleAdd computes dst[i] = src[i]*scale + bias for n elements via
// vDSP, fusing the scale and bias into a single pass over the data.
func VecScaleAdd(dst, src uintptr, scale, bias float64, n int) {
	if n <= 0 {
		return
	}
	C.af_vsmsa(C.uintptr_t(src), C.double(scale), C.double(bias), C.uintptr_t(dst), C.long(n))
}

// PoissonSolver wraps a Cholesky factorization of the pressure Poisson
// operator. The factorization is computed once per grid geometry and reused
// for every timestep; only the triangular solves run per step.
//
// The matrix arrays are borrowed from the arena, not copied. The caller
// must keep that arena alive for the solver's whole lifetime -- Close does
// not free them, it only releases Accelerate's internal factorization.
type PoissonSolver struct {
	c C.AF_Poisson
	n int
}

// NewPoissonSolver factorizes a symmetric Poisson operator supplied in
// compressed-sparse-column form, with only the lower triangle stored.
//
// columnStarts holds n+1 int64 entries, rowIndices holds nnz int32 entries,
// and values holds nnz float64 entries. All three must be arena-resident;
// passing heap memory here would both violate the zero-heap invariant and
// risk the GC moving the arrays out from under Accelerate.
func NewPoissonSolver(columnStarts, rowIndices, values uintptr, n, nnz int) (*PoissonSolver, error) {
	if n <= 0 {
		return nil, fmt.Errorf("aeroflux: poisson order must be positive, got %d", n)
	}
	if nnz <= 0 {
		return nil, fmt.Errorf("aeroflux: poisson nnz must be positive, got %d", nnz)
	}

	s := &PoissonSolver{n: n}
	rc := C.AF_PoissonFactor(
		&s.c,
		C.uintptr_t(columnStarts),
		C.uintptr_t(rowIndices),
		C.uintptr_t(values),
		C.int(n),
		C.int(nnz),
	)
	if rc != 0 {
		return nil, fmt.Errorf("aeroflux: SparseFactor failed with status %d (%s)", int(rc), sparseStatusString(int(rc)))
	}
	return s, nil
}

// SolveInPlace solves A x = rhs, overwriting rhs with x. rhs must point at
// n contiguous arena-resident float64s. No buffer is allocated: this is
// callable from the hot path without breaking the zero-allocation
// invariant.
func (s *PoissonSolver) SolveInPlace(rhs uintptr) error {
	if rc := C.AF_PoissonSolve(&s.c, C.uintptr_t(rhs), C.int(s.n)); rc != 0 {
		return fmt.Errorf("aeroflux: sparse solve on an invalid factorization")
	}
	return nil
}

// Close releases Accelerate's factorization. It does not touch the arena
// arrays the matrix borrowed.
func (s *PoissonSolver) Close() {
	C.AF_PoissonFree(&s.c)
}

// SetClusterAffinity hints that the calling thread should share a cache
// cluster with other threads carrying the same tag. Advisory only; see
// PinWorker in sys_darwin.go for why this cannot pin to a core.
//
// Call only from a goroutine already locked to its OS thread, since the
// hint applies to the current thread.
func SetClusterAffinity(tag int32) error {
	if kr := C.AF_SetClusterAffinity(C.int32_t(tag)); kr != 0 {
		return fmt.Errorf("aeroflux: thread_policy_set(THREAD_AFFINITY_POLICY) returned kern_return %d", int(kr))
	}
	return nil
}

func sparseStatusString(st int) string {
	switch st {
	case 0:
		return "SparseStatusOK"
	case -1:
		return "SparseFactorizationFailed"
	case -2:
		return "SparseMatrixIsSingular"
	case -3:
		return "SparseInternalError"
	case -4:
		return "SparseParameterError"
	default:
		return "unknown"
	}
}

// Compile-time proof that the element sizes this file assumes when building
// CSC arrays in the arena match what Accelerate's C structs actually use.
// A mismatch here (long being 4 bytes, say) would misread the entire matrix
// structure, so it is caught at build time rather than as wrong physics.
const (
	_ = uint(unsafe.Sizeof(C.long(0))) - 8
	_ = 8 - uint(unsafe.Sizeof(C.long(0)))
	_ = uint(unsafe.Sizeof(C.int(0))) - 4
	_ = 4 - uint(unsafe.Sizeof(C.int(0)))
	_ = uint(unsafe.Sizeof(C.double(0))) - 8
	_ = 8 - uint(unsafe.Sizeof(C.double(0)))
)
