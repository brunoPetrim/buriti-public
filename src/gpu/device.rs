//! GPU device layer (task G2, D63): a wgpu device and queue for one chosen adapter, buffer helpers with a
//! pool, a compiled-pipeline cache, a timed dispatch helper and error capture.
//!
//! Nothing here panics on purpose: every failure is a `Result<_, String>`. wgpu's default handler for
//! uncaptured errors panics, so this module installs its own (errors are stored and reported by the next
//! call that checks health) and records "device lost". Error scopes (thread local) wrap the operations that
//! can fail in a way the caller must see right away: shader compilation, pipeline creation and encoding.
//!
//! Features are optional: SHADER_F16, SUBGROUP and TIMESTAMP_QUERY are requested only if the adapter has
//! them; code must keep working without them (see `GpuDevice::features`).

#![allow(dead_code)] // the layer is consumed by the next tasks (G4+); parts are only used by tests and the benchmark today

use super::discovery::{enumerate_wgpu, AdapterReport};
use super::select::{select_adapter, AdapterSelector};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default wait for a GPU job or a readback. Hard limit: a hung driver becomes an error, not a hang.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// What the opened device can use (taken from the adapter, which is what was requested).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Caps {
    pub shader_f16: bool,
    pub subgroup: bool,
    pub timestamp_query: bool,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_storage_size: u32,
}

#[derive(Default)]
struct Health {
    errors: Mutex<Vec<String>>,
    lost: Mutex<Option<String>>,
}

impl Health {
    /// Err with everything captured since the last check (and the device-lost reason, which stays set).
    fn check(&self) -> Result<(), String> {
        if let Some(l) = self.lost.lock().map_err(|_| "GPU health lock poisoned")?.clone() {
            return Err(format!("GPU device lost: {l}"));
        }
        let mut e = self.errors.lock().map_err(|_| "GPU health lock poisoned")?;
        if e.is_empty() {
            Ok(())
        } else {
            let msg = e.join("; ");
            e.clear();
            Err(format!("GPU error: {msg}"))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum PoolKind {
    Storage,
    Staging,
}

#[derive(Default)]
struct Pool {
    free: HashMap<(PoolKind, u64), Vec<wgpu::Buffer>>,
    retained: u64,
}

/// A storage buffer. `len` is the logical size in bytes (what shaders see through `arrayLength`); the
/// underlying wgpu buffer may be bigger when it came from the pool.
pub struct GpuBuffer {
    pub buffer: wgpu::Buffer,
    pub len: u64,
    pooled: bool,
}

/// Pool statistics (for tests and diagnostics).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PoolStats {
    pub created: u64,
    pub reused: u64,
    pub retained_bytes: u64,
}

/// Result of one `run`.
#[derive(Debug, Clone, Copy)]
pub struct RunTiming {
    /// GPU time between the start and the end of the pass (timestamp queries), when supported.
    pub gpu_seconds: Option<f64>,
    /// Wall clock from `queue.submit` to the completion of the poll (includes driver overhead).
    pub wall_seconds: f64,
}

/// One dispatch of `run_steps`: the pipeline, the buffers bound to group 0 (bindings 0..) and the grid.
pub struct Step<'a> {
    pub pipeline: &'a wgpu::ComputePipeline,
    pub buffers: Vec<&'a GpuBuffer>,
    pub groups: [u32; 3],
}

pub struct GpuDevice {
    _instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub report: AdapterReport,
    pub caps: Caps,
    health: Arc<Health>,
    pipelines: Mutex<HashMap<String, (u64, Arc<wgpu::ComputePipeline>)>>,
    compiles: AtomicU64,
    pool: Mutex<Pool>,
    pool_created: AtomicU64,
    pool_reused: AtomicU64,
    pool_cap_bytes: u64,
}

fn hash_text(s: &str) -> u64 {
    // FNV-1a: only used to notice that the source behind a cache key changed.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn pool_bucket(len: u64, max: u64) -> u64 {
    let b = if len <= (16 << 20) { len.max(256).next_power_of_two() } else { len.div_ceil(1 << 20) << 20 };
    b.min(max).max(len)
}

fn err_text(e: wgpu::Error) -> String {
    chain(&e)
}

impl GpuDevice {
    /// Opens the adapter that matches `sel`. Never panics (a panic inside wgpu becomes an `Err`).
    pub fn open(sel: &AdapterSelector) -> Result<GpuDevice, String> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::open_inner(sel))) {
            Ok(r) => r,
            Err(_) => Err("the GPU library panicked while opening the device".into()),
        }
    }

    fn open_inner(sel: &AdapterSelector) -> Result<GpuDevice, String> {
        let (instance, mut list) = enumerate_wgpu();
        let reports: Vec<AdapterReport> = list.iter().map(|(_, r)| r.clone()).collect();
        let idx = select_adapter(&reports, sel)?;
        let (adapter, report) = list.swap_remove(idx);

        let avail = adapter.features();
        let mut features = wgpu::Features::empty();
        for f in [wgpu::Features::SHADER_F16, wgpu::Features::SUBGROUP, wgpu::Features::TIMESTAMP_QUERY] {
            if avail.contains(f) {
                features |= f;
            }
        }
        // The adapter's own limits: the largest buffers, workgroup sizes and shared memory it offers.
        let limits = adapter.limits();
        let desc = wgpu::DeviceDescriptor {
            label: Some("buriti-gpu"),
            required_features: features,
            required_limits: limits.clone(),
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).map_err(|e| format!("could not create a device on '{}' ({}): {e}", report.name, report.backend))?;

        let health = Arc::new(Health::default());
        let h = health.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = h.errors.lock() {
                if v.len() < 16 {
                    v.push(err_text(e));
                }
            }
        }));
        let h = health.clone();
        device.set_device_lost_callback(move |reason, msg| {
            // Dropping the device reports `Destroyed`: not an error worth keeping.
            if format!("{reason:?}") != "Destroyed" {
                if let Ok(mut l) = h.lost.lock() {
                    *l = Some(format!("{reason:?}: {msg}"));
                }
            }
        });

        let caps = Caps {
            shader_f16: features.contains(wgpu::Features::SHADER_F16),
            subgroup: features.contains(wgpu::Features::SUBGROUP),
            timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
        };
        Ok(GpuDevice {
            _instance: instance,
            device,
            queue,
            report,
            caps,
            health,
            pipelines: Mutex::new(HashMap::new()),
            compiles: AtomicU64::new(0),
            pool: Mutex::new(Pool::default()),
            pool_created: AtomicU64::new(0),
            pool_reused: AtomicU64::new(0),
            pool_cap_bytes: 1 << 30,
        })
    }

    pub fn name(&self) -> &str {
        &self.report.name
    }

    /// Errors captured since the last check, or "device lost". Call after any GPU work whose result matters.
    pub fn check_health(&self) -> Result<(), String> {
        self.health.check()
    }

    // ---------------------------------------------------------------- buffers

    fn check_size(&self, len: u64, what: &str) -> Result<(), String> {
        if len == 0 {
            return Err(format!("{what}: size 0"));
        }
        if len > self.caps.max_buffer_size {
            return Err(format!("{what}: {len} bytes exceed the adapter's max_buffer_size ({} bytes); splitting is not supported here", self.caps.max_buffer_size));
        }
        Ok(())
    }

    fn make_buffer(&self, size: u64, usage: wgpu::BufferUsages, label: &str) -> Result<wgpu::Buffer, String> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let scope_oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let oom = pollster::block_on(scope_oom.pop());
        let val = pollster::block_on(scope.pop());
        if let Some(e) = oom {
            return Err(format!("{label}: could not allocate {size} bytes: {}", chain(&e)));
        }
        if let Some(e) = val {
            return Err(format!("{label}: invalid buffer ({size} bytes): {}", chain(&e)));
        }
        Ok(b)
    }

    const STORAGE_USAGE: wgpu::BufferUsages =
        wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);
    const STAGING_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

    fn check_binding(&self, len: u64, what: &str) -> Result<(), String> {
        if len > self.caps.max_storage_buffer_binding_size {
            return Err(format!(
                "{what}: {len} bytes exceed max_storage_buffer_binding_size ({} bytes); splitting is not supported here",
                self.caps.max_storage_buffer_binding_size
            ));
        }
        Ok(())
    }

    /// A storage buffer of exactly `len` bytes (rounded up to a multiple of 4), not pooled. Zero-initialized.
    pub fn create_storage(&self, len: u64, label: &str) -> Result<GpuBuffer, String> {
        let len = len.div_ceil(4) * 4;
        self.check_size(len, label)?;
        self.check_binding(len, label)?;
        let buffer = self.make_buffer(len, Self::STORAGE_USAGE, label)?;
        self.pool_created.fetch_add(1, Ordering::Relaxed);
        Ok(GpuBuffer { buffer, len, pooled: false })
    }

    /// A storage buffer from the pool (contents are NOT cleared). Give it back with `release`.
    pub fn acquire_storage(&self, len: u64, label: &str) -> Result<GpuBuffer, String> {
        let len = len.div_ceil(4) * 4;
        self.check_size(len, label)?;
        self.check_binding(len, label)?;
        let size = pool_bucket(len, self.caps.max_buffer_size.min(self.caps.max_storage_buffer_binding_size));
        let buffer = self.pool_take(PoolKind::Storage, size, label)?;
        Ok(GpuBuffer { buffer, len, pooled: true })
    }

    /// Returns a pooled buffer to the pool (a non-pooled one is simply dropped).
    pub fn release(&self, b: GpuBuffer) {
        if b.pooled {
            self.pool_give(PoolKind::Storage, b.buffer);
        }
    }

    fn pool_take(&self, kind: PoolKind, size: u64, label: &str) -> Result<wgpu::Buffer, String> {
        if let Ok(mut p) = self.pool.lock() {
            if let Some(b) = p.free.get_mut(&(kind, size)).and_then(|v| v.pop()) {
                p.retained = p.retained.saturating_sub(size);
                self.pool_reused.fetch_add(1, Ordering::Relaxed);
                return Ok(b);
            }
        }
        let usage = if kind == PoolKind::Storage { Self::STORAGE_USAGE } else { Self::STAGING_USAGE };
        let b = self.make_buffer(size, usage, label)?;
        self.pool_created.fetch_add(1, Ordering::Relaxed);
        Ok(b)
    }

    fn pool_give(&self, kind: PoolKind, b: wgpu::Buffer) {
        let size = b.size();
        if let Ok(mut p) = self.pool.lock() {
            if p.retained + size <= self.pool_cap_bytes {
                p.retained += size;
                p.free.entry((kind, size)).or_default().push(b);
            }
            // else: dropped here, the memory goes back to the driver
        }
    }

    pub fn pool_stats(&self) -> PoolStats {
        let retained = self.pool.lock().map(|p| p.retained).unwrap_or(0);
        PoolStats { created: self.pool_created.load(Ordering::Relaxed), reused: self.pool_reused.load(Ordering::Relaxed), retained_bytes: retained }
    }

    /// Drops every retained buffer.
    pub fn clear_pool(&self) {
        if let Ok(mut p) = self.pool.lock() {
            p.free.clear();
            p.retained = 0;
        }
    }

    /// Writes `data` at `offset` (both multiples of 4) into the buffer.
    pub fn upload_bytes(&self, dst: &GpuBuffer, offset: u64, data: &[u8]) -> Result<(), String> {
        if offset % 4 != 0 || data.len() % 4 != 0 {
            return Err("upload: offset and length must be multiples of 4 bytes".into());
        }
        let end = offset.checked_add(data.len() as u64).ok_or("upload: offset overflow")?;
        if end > dst.len {
            return Err(format!("upload: {end} bytes do not fit in a buffer of {} bytes", dst.len));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.queue.write_buffer(&dst.buffer, offset, data);
        self.check_health()
    }

    pub fn upload_f32(&self, dst: &GpuBuffer, offset: u64, data: &[f32]) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.upload_bytes(dst, offset, &bytes)
    }

    /// New (non-pooled) storage buffer holding `data`.
    pub fn storage_from_bytes(&self, data: &[u8], label: &str) -> Result<GpuBuffer, String> {
        let b = self.create_storage(data.len() as u64, label)?;
        if data.len() % 4 == 0 {
            self.upload_bytes(&b, 0, data)?;
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().div_ceil(4) * 4, 0);
            self.upload_bytes(&b, 0, &padded)?;
        }
        Ok(b)
    }

    pub fn storage_from_f32(&self, data: &[f32], label: &str) -> Result<GpuBuffer, String> {
        let b = self.create_storage(data.len() as u64 * 4, label)?;
        self.upload_f32(&b, 0, data)?;
        Ok(b)
    }

    /// Waits for submitted work (all of it) with a hard timeout.
    fn wait_idle(&self, timeout: Duration) -> Result<(), String> {
        match self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(timeout) }) {
            Ok(_) => Ok(()),
            Err(wgpu::PollError::Timeout) => Err(format!("the GPU did not finish within {:.0} s (hang?)", timeout.as_secs_f64())),
            Err(e) => Err(format!("waiting for the GPU failed: {e}")),
        }
    }

    /// Maps a staging buffer for reading and copies `len` bytes out of it. Poll loop with a hard timeout.
    fn read_staging(&self, staging: &wgpu::Buffer, len: u64, timeout: Duration) -> Result<Vec<u8>, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..len).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let start = Instant::now();
        loop {
            match self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_millis(100)) }) {
                Ok(_) | Err(wgpu::PollError::Timeout) => {}
                Err(e) => return Err(format!("polling the GPU failed: {e}")),
            }
            match rx.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(e)) => {
                    self.check_health()?;
                    return Err(format!("mapping the readback buffer failed: {e}"));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if start.elapsed() > timeout {
                        return Err(format!("readback did not finish within {:.0} s (hang?)", timeout.as_secs_f64()));
                    }
                    self.check_health()?;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return Err("readback callback was dropped".into()),
            }
        }
        let out = {
            let view = staging.slice(..len).get_mapped_range().map_err(|e| format!("reading the mapped buffer failed: {e}"))?;
            view.to_vec()
        };
        staging.unmap();
        Ok(out)
    }

    /// Copies `len` bytes from `src` (at `offset`, multiples of 4) to the CPU: copy to a pooled staging
    /// buffer, `map_async` + poll with `timeout`.
    pub fn read_back(&self, src: &GpuBuffer, offset: u64, len: u64, timeout: Duration) -> Result<Vec<u8>, String> {
        if offset % 4 != 0 || len % 4 != 0 || len == 0 {
            return Err("readback: offset and length must be multiples of 4 bytes (length > 0)".into());
        }
        if offset.checked_add(len).is_none_or(|e| e > src.len) {
            return Err(format!("readback: range {offset}+{len} is outside the buffer ({} bytes)", src.len));
        }
        self.check_size(len, "readback")?;
        let size = pool_bucket(len, self.caps.max_buffer_size);
        let staging = self.pool_take(PoolKind::Staging, size, "readback-staging")?;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(&src.buffer, offset, &staging, 0, len);
        self.queue.submit(Some(enc.finish()));
        let r = self.read_staging(&staging, len, timeout);
        if r.is_ok() {
            self.pool_give(PoolKind::Staging, staging);
        } // after a failure the buffer may still be mapped or in flight: drop it
        let bytes = r?;
        self.check_health()?;
        Ok(bytes)
    }

    pub fn read_back_f32(&self, src: &GpuBuffer, count: usize, timeout: Duration) -> Result<Vec<f32>, String> {
        let bytes = self.read_back(src, 0, count as u64 * 4, timeout)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    // ---------------------------------------------------------------- pipelines

    /// Compiles a WGSL compute shader once per `key` and returns the cached pipeline afterwards. If the same
    /// key comes with different source text it is recompiled. Compile and validation errors (with the
    /// compiler message) are returned as `Err`.
    pub fn pipeline(&self, key: &str, wgsl: &str, entry: &str) -> Result<Arc<wgpu::ComputePipeline>, String> {
        let h = hash_text(wgsl) ^ hash_text(entry).rotate_left(17);
        if let Ok(m) = self.pipelines.lock() {
            if let Some((sh, p)) = m.get(key) {
                if *sh == h {
                    return Ok(p.clone());
                }
            }
        }
        let s1 = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let s2 = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(key), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let internal = pollster::block_on(s2.pop());
        let validation = pollster::block_on(s1.pop());
        if let Some(e) = validation {
            return Err(format!("shader '{key}' failed to compile: {}", chain(&e)));
        }
        if let Some(e) = internal {
            return Err(format!("shader '{key}' failed in the driver: {}", chain(&e)));
        }
        self.compiles.fetch_add(1, Ordering::Relaxed);
        let p = Arc::new(pipeline);
        if let Ok(mut m) = self.pipelines.lock() {
            m.insert(key.to_string(), (h, p.clone()));
        }
        Ok(p)
    }

    /// How many shaders were actually compiled (cache misses).
    pub fn compile_count(&self) -> u64 {
        self.compiles.load(Ordering::Relaxed)
    }

    // ---------------------------------------------------------------- dispatch

    /// Runs `pipeline` with `repeat` back-to-back dispatches of `groups` workgroups inside one compute pass,
    /// binding `buffers` to group 0, bindings 0.. (each with its logical length). Waits with `timeout`.
    pub fn run(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&GpuBuffer], groups: [u32; 3], repeat: u32, timeout: Duration) -> Result<RunTiming, String> {
        self.run_steps(&[Step { pipeline, buffers: buffers.to_vec(), groups }], repeat, timeout)
    }

    /// Like `run`, for a sequence of dispatches (each with its own pipeline, buffers and grid) inside ONE
    /// compute pass, the whole sequence repeated `repeat` times. Timing covers the whole pass.
    pub fn run_steps(&self, steps: &[Step], repeat: u32, timeout: Duration) -> Result<RunTiming, String> {
        let maxg = self.caps.max_compute_workgroups_per_dimension;
        if steps.is_empty() {
            return Err("dispatch: no steps".into());
        }
        for st in steps {
            if st.groups.iter().any(|&g| g == 0 || g > maxg) {
                return Err(format!("dispatch of {:?} workgroups is outside 1..={maxg} per dimension", st.groups));
            }
        }
        if repeat == 0 {
            return Err("dispatch: repeat must be at least 1".into());
        }
        let use_ts = self.caps.timestamp_query;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let binds: Vec<wgpu::BindGroup> = steps
            .iter()
            .map(|st| {
                let layout = st.pipeline.get_bind_group_layout(0);
                let entries: Vec<wgpu::BindGroupEntry> = st
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &b.buffer, offset: 0, size: std::num::NonZeroU64::new(b.len) }),
                    })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("run"), layout: &layout, entries: &entries })
            })
            .collect();
        let qs = use_ts.then(|| self.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("ts"), ty: wgpu::QueryType::Timestamp, count: 2 }));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("run") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("run"),
                timestamp_writes: qs.as_ref().map(|q| wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }),
            });
            for _ in 0..repeat {
                for (st, bind) in steps.iter().zip(&binds) {
                    pass.set_pipeline(st.pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(st.groups[0], st.groups[1], st.groups[2]);
                }
            }
        }
        let ts_bufs = match &qs {
            Some(q) => {
                let resolve = self.make_buffer(16, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, "ts-resolve")?;
                let stage = self.make_buffer(16, Self::STAGING_USAGE, "ts-stage")?;
                enc.resolve_query_set(q, 0..2, &resolve, 0);
                enc.copy_buffer_to_buffer(&resolve, 0, &stage, 0, 16);
                Some(stage)
            }
            None => None,
        };
        let cmd = enc.finish();
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("invalid dispatch: {}", chain(&e)));
        }
        let t0 = Instant::now();
        self.queue.submit(Some(cmd));
        self.wait_idle(timeout)?;
        let wall = t0.elapsed().as_secs_f64();
        self.check_health()?;
        let gpu_seconds = match ts_bufs {
            Some(stage) => {
                let raw = self.read_staging(&stage, 16, timeout)?;
                let t = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
                let (a, b) = (t(0), t(8));
                let period = self.queue.get_timestamp_period() as f64; // ns per tick
                (b > a).then(|| (b - a) as f64 * period * 1e-9)
            }
            None => None,
        };
        Ok(RunTiming { gpu_seconds, wall_seconds: wall })
    }
}

/// Message of an error with its chain of causes (wgpu's `Display` alone is just "Validation Error").
fn chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(c) = cur {
        let t = c.to_string();
        if !s.contains(&t) {
            s.push_str(": ");
            s.push_str(&t);
        }
        cur = c.source();
    }
    s
}
