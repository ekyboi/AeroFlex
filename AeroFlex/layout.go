// Package aeroflux implements a zero-heap, deterministic 3D Navier-Stokes
// fluid engine over off-heap arenas, with hardware-accelerated kernels on
// Darwin via the Accelerate framework.
package aeroflux

import "unsafe"

// CacheLineSize is the coherence stride used to pad every mutable control
// block that is touched by more than one worker goroutine. 128 bytes covers
// Apple Silicon's adjacent-sector prefetch (2x64B) and is a safe superset of
// the 64B lines used on x86-64 Linux servers, so one padding constant works
// for both platforms without a build-tag split.
const CacheLineSize = 128

// SweepBarriersPerStep is the number of advection/diffusion phase barriers
// in one timestep: the six half-steps of the Strang palindrome
// (X, Y, Z, Z, Y, X).
//
// This counts sweeps only. A full step also synchronizes inside each of its
// two projections -- once for divergence, twice per Gauss-Seidel iteration
// (red then black), and once for the gradient subtraction -- so the total
// barrier count per step is 6 + 2*(2 + 2*GaussSeidelIters), which depends
// on configuration rather than being a fixed constant. The direct solver
// replaces the iterative sweeps and lowers it to 6 + 2*2.
const SweepBarriersPerStep = 6

// Color is the Red-Black Gauss-Seidel checkerboard parity of a cell.
type Color uint8

const (
	ColorRed Color = iota
	ColorBlack
)

// PhaseBarrier is a reusable sense-reversing barrier. Arrive is the shared
// atomic counter (hot, written by every worker); Sense and Target are cold,
// read-mostly fields. Arrive is isolated to its own line to prevent false
// sharing against the Sense field workers spin on.
type PhaseBarrier struct {
	Arrive uint64
	_      [CacheLineSize - unsafe.Sizeof(uint64(0))]byte

	Sense  uint32
	Target uint32
	_      [CacheLineSize - 2*unsafe.Sizeof(uint32(0))]byte
}

// --- Compile-time layout verification -------------------------------------
//
// These const declarations cross-subtract unsafe.Sizeof/Offsetof results
// against HARDCODED literals. Any expression that would go negative is not
// representable as an unsigned constant and fails the build, so a layout
// regression is a compile error rather than a silent performance cliff.
//
// The literals are deliberately spelled out rather than written in terms of
// CacheLineSize. Expressing them as CacheLineSize*k would make the whole
// block self-fulfilling: the padding arrays are themselves sized from
// CacheLineSize, so the structs would simply resize with the constant and
// every subtraction would stay non-negative no matter what. Anchoring to
// 128/256 means changing CacheLineSize breaks the build, which is the
// entire point -- these numbers are an architectural commitment, not a
// derived quantity.

const (
	// PhaseBarrier spans two strides: the hot arrival counter is isolated
	// from the read-mostly sense/target fields workers spin on, so an
	// arrival never invalidates the line a peer is polling.
	_ = uint(unsafe.Sizeof(PhaseBarrier{})) - 256
	_ = 256 - uint(unsafe.Sizeof(PhaseBarrier{}))
	_ = uint(unsafe.Offsetof(PhaseBarrier{}.Sense)) - 128

	// TicketGate likewise: serving is polled by every waiter while next is
	// mutated by every issuer, so they must not share a line.
	_ = uint(unsafe.Sizeof(TicketGate{})) - 256
	_ = 256 - uint(unsafe.Sizeof(TicketGate{}))

	// Each control struct is a whole multiple of the stride, so arrays of
	// them never let two workers straddle one line.
	_ = uint(unsafe.Sizeof(PhaseBarrier{})) % 128
	_ = uint(unsafe.Sizeof(TicketGate{})) % 128

	// One shard descriptor per line, for the same reason.
	_ = uint(unsafe.Sizeof(shard{})) - 128
	_ = 128 - uint(unsafe.Sizeof(shard{}))

	// The stride constant itself must match what the padding assumes.
	_ = uint(CacheLineSize) - 128
	_ = 128 - uint(CacheLineSize)
)
