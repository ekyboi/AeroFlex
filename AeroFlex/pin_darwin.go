//go:build darwin

package aeroflux

// pinWorkerForShard applies Darwin thread policy for worker id.
//
// Each worker gets a distinct nonzero affinity tag so the scheduler is
// hinted to keep it on its own L2 cluster rather than migrating it around.
// Both calls are advisory and best-effort: macOS exposes no hard core
// pinning, so a failure here costs scheduling quality, never correctness.
func pinWorkerForShard(id int) {
	_ = PinWorker()
	_ = SetClusterAffinity(int32(id + 1))
}
