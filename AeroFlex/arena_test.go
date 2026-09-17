package aeroflux

import (
	"math"
	"runtime/debug"
	"testing"
	"unsafe"
)

func TestArenaAlignmentAndGuards(t *testing.T) {
	a, err := NewArena(4 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	if a.Base%HugePageSize != 0 {
		t.Errorf("Base %#x is not %d-aligned", a.Base, HugePageSize)
	}
	if a.Len%HugePageSize != 0 {
		t.Errorf("Len %d is not a multiple of %d", a.Len, HugePageSize)
	}
	if a.Len < 4<<20 {
		t.Errorf("Len %d smaller than requested", a.Len)
	}
	t.Logf("base=%#x len=%d pagesize=%d locked=%v", a.Base, a.Len, PageSize, a.Locked())
}

func TestArenaReadWrite(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	n := 4096
	xs := a.Float64s(0, n)
	for i := range xs {
		xs[i] = float64(i) * 0.5
	}
	for i := range xs {
		if xs[i] != float64(i)*0.5 {
			t.Fatalf("readback mismatch at %d: got %v", i, xs[i])
		}
	}
}

// TestGuardPageFaultIsContained is the load-bearing safety test: an access
// past the end of the usable region must fault and be recoverable, not
// corrupt adjacent memory and not kill the process.
func TestGuardPageFaultIsContained(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	done := make(chan error, 1)
	go func() {
		debug.SetPanicOnFault(true)
		var ferr error
		defer func() { done <- ferr }()
		defer RecoverFault(&ferr)

		// One byte past the end of the usable region is the trailing
		// PROT_NONE guard page.
		p := (*byte)(unsafe.Pointer(a.Base + uintptr(a.Len)))
		*p = 1 // must fault

		ferr = nil
	}()

	err = <-done
	if err == nil {
		t.Fatal("expected a contained guard-page fault, got nil -- the guard page is not protecting the region")
	}
	t.Logf("contained as expected: %v", err)
}

// TestGuardPageBelowBase covers the leading guard page.
func TestGuardPageBelowBase(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	done := make(chan error, 1)
	go func() {
		debug.SetPanicOnFault(true)
		var ferr error
		defer func() { done <- ferr }()
		defer RecoverFault(&ferr)

		p := (*byte)(unsafe.Pointer(a.Base - 1))
		_ = *p // must fault

		ferr = nil
	}()

	if err := <-done; err == nil {
		t.Fatal("expected a contained fault below Base")
	}
}

func TestArenaCloseIdempotent(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close should be a no-op: %v", err)
	}
}

// TestHotPathZeroAllocs asserts the zero-Go-heap-allocation invariant on a
// representative hot-path operation: a strided write sweep plus an
// Accelerate vForce call over arena memory.
func TestHotPathZeroAllocs(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const n = 8192
	src := a.Pointer(0)
	dst := a.Pointer(n * 8)

	xs := a.Float64s(0, n)
	for i := range xs {
		xs[i] = float64(i) * 1e-3
	}

	got := testing.AllocsPerRun(100, func() {
		VecSin(dst, src, n)
		VecCos(dst, src, n)
	})
	if got != 0 {
		t.Errorf("hot path allocated %v times per run, want 0", got)
	}
}

// TestVForceCorrectness checks that the vForce shims compute what they
// claim over raw arena pointers -- specifically that the by-pointer count
// convention was wired correctly, which is the easy thing to get silently
// wrong.
func TestVForceCorrectness(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const n = 1024
	srcOff, dstOff := 0, n*8

	src := a.Float64s(srcOff, n)
	dst := a.Float64s(dstOff, n)
	for i := range src {
		src[i] = float64(i) * 0.01
	}
	for i := range dst {
		dst[i] = math.NaN()
	}

	VecSin(a.Pointer(dstOff), a.Pointer(srcOff), n)

	for i := 0; i < n; i++ {
		want := math.Sin(src[i])
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("vvsin mismatch at %d: got %v want %v", i, dst[i], want)
		}
	}

	VecExp(a.Pointer(dstOff), a.Pointer(srcOff), n)
	for i := 0; i < n; i++ {
		want := math.Exp(src[i])
		if math.Abs(dst[i]-want) > 1e-9*math.Abs(want)+1e-12 {
			t.Fatalf("vvexp mismatch at %d: got %v want %v", i, dst[i], want)
		}
	}
}

func TestVecScaleAdd(t *testing.T) {
	a, err := NewArena(2 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const n = 512
	srcOff, dstOff := 0, n*8
	src := a.Float64s(srcOff, n)
	dst := a.Float64s(dstOff, n)
	for i := range src {
		src[i] = float64(i)
	}

	VecScaleAdd(a.Pointer(dstOff), a.Pointer(srcOff), 2.5, -1.0, n)

	for i := 0; i < n; i++ {
		want := src[i]*2.5 - 1.0
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("vsmsa mismatch at %d: got %v want %v", i, dst[i], want)
		}
	}
}

// TestGridStrides pins the memory layout the prefetcher depends on: X must
// be unit-stride, with Y and Z derived from the padded extents.
func TestGridStrides(t *testing.T) {
	a, err := NewArena(MACBytes(64, 32, 16, 1))
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	m, err := NewMAC(a, 64, 32, 16, 1, 1.0/64)
	if err != nil {
		t.Fatalf("NewMAC: %v", err)
	}

	// Cell-centered pressure: padded extents are interior + 2*ghost.
	if got := m.P.Index(1, 0, 0) - m.P.Index(0, 0, 0); got != 1 {
		t.Errorf("X stride = %d, want 1 (unit stride is required for prefetch)", got)
	}
	if m.P.SY != 66 {
		t.Errorf("Y stride = %d, want 66", m.P.SY)
	}
	if m.P.SZ != 66*34 {
		t.Errorf("Z stride = %d, want %d", m.P.SZ, 66*34)
	}
}
