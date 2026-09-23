//! Order-preserving parallel map, used by the per-frame analysis loops.
//!
//! Frame analyses are independent once the global quantities (peaks,
//! thresholds, windows, FFT plans) are fixed, so each frame can run on any
//! thread. Results are collected in input order and each item's arithmetic
//! is unchanged, so the output is bit-identical to a serial loop regardless
//! of thread count. wasm32 has no threads and runs the same closure serially.
//!
//! Thread count: `RAYON_NUM_THREADS` (default: all cores). When called from
//! inside a caller's rayon pool (`ThreadPool::install`), that pool is used.
//!
//! Fork safety: rayon's global pool does not survive `fork()` (the child
//! inherits the pool's state but none of its worker threads, so the first
//! parallel call deadlocks). This bites Python `multiprocessing` with the
//! fork start method. We therefore use our own pool, keyed on the process
//! id, and build a fresh one in a forked child.

/// Map `f` over `0..n`, returning results in index order.
///
/// `init` creates scratch state (reusable buffers). rayon calls it once per
/// split job rather than once per thread, and the wasm32 path calls it once
/// for all items, so `f` must fully overwrite whatever scratch it reads: a
/// result must never depend on which items previously used the same scratch.
pub(crate) fn map_frames<S, T, I, F>(n: usize, init: I, f: F) -> Vec<T>
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> T + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        let run = || (0..n).into_par_iter().map_init(&init, &f).collect();
        if rayon::current_thread_index().is_some() {
            // Already on a rayon worker (caller's pool, or nested): stay in it.
            run()
        } else {
            pool().install(run)
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut scratch = init();
        (0..n).map(|i| f(&mut scratch, i)).collect()
    }
}

/// The crate's thread pool, rebuilt after `fork()`.
#[cfg(not(target_arch = "wasm32"))]
fn pool() -> std::sync::Arc<rayon::ThreadPool> {
    use std::sync::{Arc, Mutex};

    static POOL: Mutex<Option<(u32, Arc<rayon::ThreadPool>)>> = Mutex::new(None);

    let pid = std::process::id();
    let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((owner, pool)) = guard.as_ref() {
        if *owner == pid {
            return Arc::clone(pool);
        }
    }
    // First use, or we are a forked child holding the parent's pool, whose
    // workers do not exist here. Leak the stale pool: dropping it would try
    // to signal and join threads that were never forked.
    if let Some((_, stale)) = guard.take() {
        std::mem::forget(stale);
    }
    // ThreadPoolBuilder::new() honours RAYON_NUM_THREADS.
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("praatfan-{i}"))
            .build()
            .expect("failed to build praatfan thread pool"),
    );
    *guard = Some((pid, Arc::clone(&pool)));
    pool
}
