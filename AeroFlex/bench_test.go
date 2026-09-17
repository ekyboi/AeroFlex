package aeroflux

import (
	"fmt"
	"testing"
)

func BenchmarkStep(b *testing.B) {
	for _, n := range []int{16, 32, 48} {
		b.Run(fmt.Sprintf("%dx%dx%d", n, n, n), func(b *testing.B) {
			e, err := New(Config{
				NX: n, NY: n, NZ: n,
				H: 1.0 / float64(n), DT: 0.002,
				GaussSeidelIters: 12,
			})
			if err != nil {
				b.Fatalf("New: %v", err)
			}
			defer e.Close()
			seedVortex(e, 1.0)

			cells := n * n * n
			b.SetBytes(int64(cells * 8 * 4)) // u,v,w,p touched per step
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := e.Step(); err != nil {
					b.Fatalf("Step: %v", err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(cells)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mcell/s")
		})
	}
}

func BenchmarkStepByWorkers(b *testing.B) {
	const n = 32
	for _, w := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("workers=%d", w), func(b *testing.B) {
			e, err := New(Config{
				NX: n, NY: n, NZ: n,
				H: 1.0 / float64(n), DT: 0.002,
				Workers:          w,
				GaussSeidelIters: 12,
			})
			if err != nil {
				b.Fatalf("New: %v", err)
			}
			defer e.Close()
			seedVortex(e, 1.0)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := e.Step(); err != nil {
					b.Fatalf("Step: %v", err)
				}
			}
			b.StopTimer()
			cells := n * n * n
			b.ReportMetric(float64(cells)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mcell/s")
		})
	}
}

func BenchmarkProjectOnly(b *testing.B) {
	const n = 32
	e, err := New(Config{
		NX: n, NY: n, NZ: n, H: 1.0 / n, DT: 0.002,
		GaussSeidelIters: 20,
	})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer e.Close()
	seedVortex(e, 1.0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := e.project(); err != nil {
			b.Fatalf("project: %v", err)
		}
	}
}

func BenchmarkVecSin(b *testing.B) {
	a, err := NewArena(4 << 20)
	if err != nil {
		b.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const n = 65536
	src := a.Float64s(0, n)
	for i := range src {
		src[i] = float64(i) * 1e-4
	}
	sp, dp := a.Pointer(0), a.Pointer(n*8)

	b.SetBytes(int64(n * 8))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VecSin(dp, sp, n)
	}
}

func BenchmarkArenaAllocFree(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a, err := NewArena(2 << 20)
		if err != nil {
			b.Fatalf("NewArena: %v", err)
		}
		if err := a.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
	}
}
