package aeroflux

import (
	"runtime"
	"sync/atomic"

	"aeroflux/internal/cpu"
)

// spinThenYield backs off a contended spin. The first few iterations use a
// pure pause/hint so a barrier that resolves in tens of nanoseconds never
// pays a scheduler round trip; past that it yields, because continuing to
// burn a core while a peer is descheduled actively delays the peer we are
// waiting on.
func spinThenYield(iter int) {
	if iter < 64 {
		cpu.Relax()
		return
	}
	runtime.Gosched()
}

// --- Sense-reversing phase barrier ----------------------------------------

// Init prepares the barrier for n participating workers. It must be called
// before any worker reaches Wait, and never concurrently with Wait.
func (b *PhaseBarrier) Init(n int) {
	atomic.StoreUint64(&b.Arrive, 0)
	atomic.StoreUint32(&b.Target, uint32(n))
	atomic.StoreUint32(&b.Sense, 0)
}

// Wait blocks until all Target workers have arrived, then releases them all.
//
// Sense reversal is what makes the barrier reusable without a reset phase:
// each worker captures the current sense before arriving, and waits for the
// global sense to flip away from it. A worker that races ahead into the
// next barrier episode cannot pass, because the sense it captures there is
// the already-flipped value and it must wait for the following flip.
//
// The last arriver resets Arrive to 0 *before* flipping Sense. That order
// matters: the flip is the release signal, so any worker woken by it must
// already see a zeroed counter, or it could observe a stale count in the
// next episode.
func (b *PhaseBarrier) Wait() {
	mySense := atomic.LoadUint32(&b.Sense)
	target := uint64(atomic.LoadUint32(&b.Target))

	if atomic.AddUint64(&b.Arrive, 1) == target {
		atomic.StoreUint64(&b.Arrive, 0)
		// Release every spinner. The store to Sense publishes all writes
		// this worker made before the barrier.
		if mySense == 0 {
			atomic.StoreUint32(&b.Sense, 1)
		} else {
			atomic.StoreUint32(&b.Sense, 0)
		}
		return
	}

	for i := 0; atomic.LoadUint32(&b.Sense) == mySense; i++ {
		spinThenYield(i)
	}
}

// --- Position-monotonic ticket turnstile ----------------------------------

// TicketGate sequences workers through a shared region in a fixed,
// position-monotonic order. Unlike a mutex, the admission order is decided
// by Position at issue time rather than by which worker happens to win a
// race, so the sequence of operations is identical on every run regardless
// of GOMAXPROCS or scheduling. That property is what makes reductions
// through the gate bit-for-bit reproducible.
type TicketGate struct {
	serving uint64
	_       [CacheLineSize - 8]byte

	next uint64
	_    [CacheLineSize - 8]byte
}

// Reset returns the gate to its initial state. Not safe to call
// concurrently with Acquire/Release.
func (g *TicketGate) Reset() {
	atomic.StoreUint64(&g.serving, 0)
	atomic.StoreUint64(&g.next, 0)
}

// Issue hands out the next monotonic position. Positions are consumed in
// ascending order by Acquire.
func (g *TicketGate) Issue() uint64 {
	return atomic.AddUint64(&g.next, 1) - 1
}

// Acquire blocks until the gate is serving pos.
func (g *TicketGate) Acquire(pos uint64) {
	for i := 0; atomic.LoadUint64(&g.serving) != pos; i++ {
		spinThenYield(i)
	}
}

// Release advances the gate to pos+1, admitting the next holder. The
// atomic store publishes everything the caller wrote while holding the
// gate to the worker that observes the new value.
func (g *TicketGate) Release(pos uint64) {
	atomic.StoreUint64(&g.serving, pos+1)
}

// Do runs fn under the gate at position pos. Even if fn panics -- a
// contained guard-page fault, say -- the gate is still advanced, so a
// single faulting worker cannot deadlock every other worker behind it.
func (g *TicketGate) Do(pos uint64, fn func()) {
	g.Acquire(pos)
	defer g.Release(pos)
	fn()
}

// --- Deterministic ordered reduction --------------------------------------

// OrderedAccumulator combines per-worker partial sums in a fixed index
// order rather than in completion order.
//
// This exists because floating-point addition is not associative: summing
// the same partials in a different order yields different low bits. A
// conventional atomic-add accumulator therefore produces run-to-run
// variation that tracks scheduling noise. Here each worker writes only its
// own slot, and the final combine walks slots 0..n-1 in index order, so the
// result depends on the partition but never on timing.
//
// Slots are padded to CacheLineSize so concurrent writers never share a
// line.
type OrderedAccumulator struct {
	slots []accSlot
}

type accSlot struct {
	val float64
	_   [CacheLineSize - 8]byte
}

// NewOrderedAccumulator allocates n padded slots. This allocates on the Go
// heap and is therefore setup-only, never hot path.
func NewOrderedAccumulator(n int) *OrderedAccumulator {
	return &OrderedAccumulator{slots: make([]accSlot, n)}
}

// Reset zeroes every slot.
func (a *OrderedAccumulator) Reset() {
	for i := range a.slots {
		a.slots[i].val = 0
	}
}

// Set stores worker i's partial. Each worker owns exactly one index, so no
// atomic is needed; the barrier that precedes Sum provides the ordering.
func (a *OrderedAccumulator) Set(i int, v float64) {
	a.slots[i].val = v
}

// Add folds v into worker i's own partial.
func (a *OrderedAccumulator) Add(i int, v float64) {
	a.slots[i].val += v
}

// Sum combines all partials in ascending index order. Call only after a
// barrier has established that every worker's Set/Add is visible.
func (a *OrderedAccumulator) Sum() float64 {
	var total float64
	for i := range a.slots {
		total += a.slots[i].val
	}
	return total
}

// Max returns the largest partial, scanning in index order. Used for the
// CFL velocity bound, where the same determinism argument applies.
func (a *OrderedAccumulator) Max() float64 {
	m := a.slots[0].val
	for i := 1; i < len(a.slots); i++ {
		if a.slots[i].val > m {
			m = a.slots[i].val
		}
	}
	return m
}
