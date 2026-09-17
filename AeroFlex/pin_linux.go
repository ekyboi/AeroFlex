//go:build linux

package aeroflux

import "runtime"

// pinWorkerForShard binds worker id to a specific CPU.
//
// Unlike Darwin this is a hard guarantee from the kernel. Workers are
// assigned round-robin over the available CPUs; with Workers <= GOMAXPROCS
// (enforced in Config.Normalize) each worker lands on its own core.
func pinWorkerForShard(id int) {
	n := runtime.NumCPU()
	if n <= 0 {
		return
	}
	_ = PinWorkerToCPU(id % n)
}
