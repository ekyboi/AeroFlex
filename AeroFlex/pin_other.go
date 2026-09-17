//go:build !darwin && !linux

package aeroflux

import "runtime"

// pinWorkerForShard locks the goroutine to its thread on platforms without
// an affinity API wired up. The OS-thread lock alone still prevents the Go
// scheduler from migrating the goroutine mid-sweep.
func pinWorkerForShard(id int) {
	_ = id
	runtime.LockOSThread()
}
