package aeroflux

import (
	"runtime"
	"sync"
	"testing"
)

func TestPhaseBarrierReleasesAllWorkers(t *testing.T) {
	const workers, rounds = 8, 200
	var b PhaseBarrier
	b.Init(workers)

	// counter is only ever mutated between barriers by a single designated
	// worker, so a torn read here would mean the barrier failed to order
	// the phases.
	counter := 0
	seen := make([][]int, workers)
	for i := range seen {
		seen[i] = make([]int, 0, rounds)
	}

	var wg sync.WaitGroup
	wg.Add(workers)
	for id := 0; id < workers; id++ {
		go func(id int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				if id == 0 {
					counter++
				}
				b.Wait()
				seen[id] = append(seen[id], counter)
				b.Wait()
			}
		}(id)
	}
	wg.Wait()

	// Every worker must observe the same value in every round: proof the
	// barrier released them all into the same phase.
	for r := 0; r < rounds; r++ {
		want := seen[0][r]
		if want != r+1 {
			t.Fatalf("round %d: worker 0 saw %d, want %d", r, want, r+1)
		}
		for id := 1; id < workers; id++ {
			if got := seen[id][r]; got != want {
				t.Fatalf("round %d: worker %d saw %d, worker 0 saw %d", r, id, got, want)
			}
		}
	}
}

func TestPhaseBarrierIsReusable(t *testing.T) {
	var b PhaseBarrier
	b.Init(2)
	var wg sync.WaitGroup
	wg.Add(2)
	for i := 0; i < 2; i++ {
		go func() {
			defer wg.Done()
			for r := 0; r < 5000; r++ {
				b.Wait()
			}
		}()
	}
	wg.Wait() // hangs on a sense-reversal bug
}

func TestTicketGateAdmitsInPositionOrder(t *testing.T) {
	var g TicketGate
	g.Reset()

	const n = 64
	order := make([]int, 0, n)
	var mu sync.Mutex

	// Positions are issued up front, so admission order is fixed before any
	// goroutine starts racing.
	positions := make([]uint64, n)
	for i := range positions {
		positions[i] = g.Issue()
	}

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			// Stagger starts so completion order would differ from
			// position order if the gate were not enforcing it.
			if i%2 == 0 {
				runtime.Gosched()
			}
			g.Do(positions[i], func() {
				mu.Lock()
				order = append(order, i)
				mu.Unlock()
			})
		}(i)
	}
	wg.Wait()

	if len(order) != n {
		t.Fatalf("got %d entries, want %d", len(order), n)
	}
	for i, v := range order {
		if v != i {
			t.Fatalf("admission order broken at %d: got worker %d", i, v)
		}
	}
}

// TestTicketGateSurvivesPanic covers the guarantee that a contained fault
// inside a gated section still advances the gate, so peers behind it do not
// deadlock.
func TestTicketGateSurvivesPanic(t *testing.T) {
	var g TicketGate
	g.Reset()

	p0, p1 := g.Issue(), g.Issue()

	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() { _ = recover() }()
		g.Do(p0, func() { panic("simulated fault") })
	}()
	<-done

	admitted := make(chan struct{})
	go func() {
		g.Do(p1, func() { close(admitted) })
	}()
	<-admitted // blocks forever if Do did not release on panic
}

func TestOrderedAccumulatorIsOrderIndependent(t *testing.T) {
	const n = 16
	a := NewOrderedAccumulator(n)

	// Values chosen so that summing them in a different order gives a
	// different float64: a large value alongside many tiny ones loses the
	// tiny contributions if they are added to the large one first.
	vals := make([]float64, n)
	vals[0] = 1e16
	for i := 1; i < n; i++ {
		vals[i] = 1.0
	}

	// Fill sequentially.
	a.Reset()
	for i := 0; i < n; i++ {
		a.Set(i, vals[i])
	}
	seq := a.Sum()

	// Fill concurrently, i.e. in arbitrary completion order.
	a.Reset()
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			a.Set(i, vals[i])
		}(i)
	}
	wg.Wait()
	conc := a.Sum()

	if seq != conc {
		t.Errorf("sum depends on fill order: %v vs %v", seq, conc)
	}
}

func TestOrderedAccumulatorMax(t *testing.T) {
	a := NewOrderedAccumulator(4)
	a.Reset()
	a.Set(0, -5)
	a.Set(1, 3)
	a.Set(2, 11)
	a.Set(3, 7)
	if got := a.Max(); got != 11 {
		t.Errorf("Max = %v, want 11", got)
	}
}

func TestOrderedAccumulatorAdd(t *testing.T) {
	a := NewOrderedAccumulator(2)
	a.Reset()
	a.Add(0, 1.5)
	a.Add(0, 2.5)
	a.Add(1, 10)
	if got := a.Sum(); got != 14 {
		t.Errorf("Sum = %v, want 14", got)
	}
}
