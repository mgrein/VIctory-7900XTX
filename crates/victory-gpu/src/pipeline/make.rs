//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).
//!
//! A program that fills in data once for the whole run is started when the
//! title creates it and finished before its first dispatch, outside the
//! frame's budget ([`fills_in_once`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]).
    screen: bool,
    /// What first uses have waited for it so far, over all frames and
    /// threads, in microseconds ([`total_capped`]).
    waited_us: AtomicU64,
    /// Why it is never made ([`left_out`]); its draws and dispatches are
    /// dropped as those of a pipeline the host refused.
    left_out: Option<&'static str>,
    /// Its one dispatch must not be lost ([`fills_in_once`]).
    once: bool,
}

/// Whether the pipeline of `shaders` fills in data the title keeps for the
/// whole run with one dispatch: the program that maps the heads' blend
/// shapes (`CS_ComputeBlendshapeMappings`). On a cold cache its pipeline
/// was not there within the frame's budget, the dispatch was dropped, and
/// every later skinning of a head read mappings nobody had written: faces
/// stayed broken to the run's end, long after the workers had caught up.
/// Such a pipeline is started when the title creates it and, if it is still
/// not there at its first dispatch, made or waited for on that thread: one
/// compile, once. (As the cool_fox fork's 700fc784 by Xyzstk, its PR #30.)
/// On by default; `VICTORY_PIPELINE_ONCE=0`: within the frame's budget like
/// every other, as before.
fn fills_in_once(shaders: &[Arc<Shader>]) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_ONCE").is_none_or(|v| v != "0"))
        && named(shaders, b"CS_ComputeBlendshapeMappings")
}

/// Whether one of the compute programs in `shaders` is `name`: the title's
/// containers hold their programs' names.
fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders.iter().any(|s| s.stage == "compute" && holds(&s.container, name))
}

/// Whether `bytes` hold `name` (not empty) somewhere.
fn holds(bytes: &[u8], name: &[u8]) -> bool {
    let mut rest = bytes;
    while let Some(at) = rest.iter().position(|b| *b == name[0]) {
        if rest[at..].starts_with(name) {
            return true;
        }
        rest = &rest[at + 1..];
    }
    false
}

/// The ray-tracing passes switched off for a run that traces no rays:
/// `VICTORY_RT_RESOLVE=0` leaves out the programs that shade what the rays
/// hit (`CS_RaytracingResolve`, about a thousand of them: on a cold cache
/// they fill the workers' queue and the library for nothing but the answer
/// "no hit"), `VICTORY_ACTIVATE_RAYS=0` the one that picks the rays to send
/// (`CS_ScreenProbes_ActivateRays`). Both are made by default, as before:
/// without them the targets they fill keep what was in them, which changes
/// the picture. Neither switch counts in a run that builds the title's ray
/// structures (`VICTORY_BVH_BUILDER=1`) or traces natively
/// (`VICTORY_NATIVE_RT=1`): there the passes have hits to shade.
/// (After DexNom ab34a8b2, pipeline/ray_tracing.rs, and the cool_fox fork's
/// 8cc31ab3 by codec_d, `left_out`.)
fn rays_left_out() -> [bool; 2] {
    static OUT: OnceLock<[bool; 2]> = OnceLock::new();
    *OUT.get_or_init(|| {
        let off = |name: &str| std::env::var_os(name).is_some_and(|v| v == "0");
        let asked = [off("VICTORY_RT_RESOLVE"), off("VICTORY_ACTIVATE_RAYS")];
        let traced = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1") || victory_shader::emit::native_rt();
        if asked == [false, false] {
            return asked;
        }
        if traced {
            tracing::warn!("VICTORY_RT_RESOLVE=0 and VICTORY_ACTIVATE_RAYS=0 do not count in a run that traces rays");
            return [false, false];
        }
        tracing::info!(resolve = asked[0], activate = asked[1], "ray-tracing passes left out: their pipelines are not made");
        asked
    })
}

/// Why the pipeline of `shaders` is not made at all, with the passes
/// `out` ([`rays_left_out`]) left out.
fn left_out(shaders: &[Arc<Shader>], out: [bool; 2]) -> Option<&'static str> {
    if out[0] && named(shaders, b"CS_RaytracingResolve") {
        return Some("shades ray hits in a run that traces no rays (VICTORY_RT_RESOLVE=0)");
    }
    if out[1] && named(shaders, b"CS_ScreenProbes_ActivateRays") {
        return Some("picks rays in a run that traces none (VICTORY_ACTIVATE_RAYS=0)");
    }
    None
}

/// Pipelines left out so far ([`left_out`]).
static LEFT_OUT: AtomicU64 = AtomicU64::new(0);

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        let out = rays_left_out();
        // (The names are looked for only with a switch set: 84,000
        // pipelines are created while the title loads.)
        let left_out = if out == [false, false] { None } else { left_out(&shaders, out) };
        if left_out.is_some() && LEFT_OUT.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::info!(?shaders, why = left_out, "a pipeline is left out (the first; further ones are not logged)");
        }
        Pending {
            screen: graphics.as_ref().is_some_and(|g| g.screen_pass()),
            // (Compute programs alone are looked through: one shader each.)
            once: graphics.is_none() && fills_in_once(&shaders),
            left_out,
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            waited_us: AtomicU64::new(0),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.made.get_or_init(|| {
            let _building = crate::timeline::span("pipeline: build a host pipeline");
            let began = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            made
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// The hashes of the shaders of `p` translated in an approximate form
/// (`HostShader::approximate`); empty when none is (or none is translated).
pub(crate) fn approximate(p: &Pending) -> Vec<String> {
    p.shaders
        .iter()
        .filter(|s| s.translated().is_some_and(|t| t.host.approximate))
        .map(|s| s.hash.clone())
        .collect()
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        if p.once {
            // Waits for the worker that is making it, or makes it here if
            // none has taken it yet: the frame's budget cannot drop a
            // dispatch that happens once.
            let began = Instant::now();
            p.make();
            tracing::info!(
                shaders = ?p.shaders,
                ms = began.elapsed().as_millis() as u64,
                made = p.made.get().is_some_and(|made| made.is_some()),
                "a pipeline that fills in data once was finished before its first dispatch"
            );
        } else {
            wait(p, crate::frame::current());
        }
    }
    p.made.get()?.as_ref()
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if p.once {
        // Before the others, whether an earlier run used it or not.
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, REQUESTED);
    } else if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Whether one pipeline is waited for at most one budget in all, over every
/// frame it is asked for in. The frame's budget begins anew in every frame:
/// a pipeline whose compile takes three seconds took the whole half second
/// of each frame that drew with it until it was there, six frames of half a
/// second while the world streams in, and its draws were dropped in all of
/// them anyway. With the cap the first frame waits, and the later ones drop
/// its draws at once until a worker has it. (After
/// Belobug/VIctory-Belobug-AMD PR #229 by ESAT586, DexNom's and
/// fxceless/victory-fork PR #1234 and #1312.)
/// On by default; `VICTORY_PIPELINE_WAIT_TOTAL=0`: a budget in every frame,
/// as before.
fn total_capped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_TOTAL").is_none_or(|v| v != "0"))
}

/// What a first use may wait now: what is left of the frame's `budget`
/// (`spent` of it is used or taken), and with `capped` no more than what is
/// left of one budget for this pipeline, `waited` for already.
fn allowance(budget: Duration, spent: Duration, waited: Duration, capped: bool) -> Duration {
    let left = budget.saturating_sub(spent);
    if capped { left.min(budget.saturating_sub(waited)) } else { left }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    let capped = total_capped();
    let waited_before = Duration::from_micros(p.waited_us.load(Ordering::Relaxed));
    if capped && waited_before >= budget {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let shared = shared_budget();
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = allowance(budget, spent.1, waited_before, capped);
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        return;
    }
    let waiting = crate::timeline::span("pipeline: first use waits for a host pipeline");
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    p.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    waiting.end();
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (worker_count(None, std::thread::available_parallelism().map_or(2, |n| n.get()), total_ram()) / 4).max(1)
        }
    })
}

/// Whether the workers' queues are ordered by what a pipeline draws. Among
/// those a draw waits for, the ones with a vertex shader (the passes that
/// put the picture together are among them) go before compute and mesh
/// pipelines, which on a cold cache are many and slow; among those started
/// early, the ones that write depth alone (shadow maps, depth passes) go
/// first: several later draws read what they drew, so one of them missing
/// costs more of the picture than another. Each kind in the order asked for.
/// (After VItoriaBr 8cb509bc by uiliannn and DexNom 921d8d3b.)
/// On by default; `VICTORY_PIPELINE_ORDER=0`: in the order asked for alone,
/// as before.
fn ordered() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_ORDER").is_none_or(|v| v != "0"))
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    /// Requested ones with a vertex shader ([`ordered`]): before the other
    /// requested ones.
    requested_vertex: VecDeque<Arc<Pending>>,
    /// Early ones that write depth alone ([`ordered`]): before the other
    /// early ones.
    early_depth: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            requested_vertex: VecDeque::new(),
            early_depth: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        self.push_as(p, priority, slow_limit() != usize::MAX && super::library::slow(&p.key))
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        self.push_in(p, priority, slow, ordered())
    }

    fn push_in(&mut self, p: &Arc<Pending>, priority: u8, slow: bool, ordered: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED && ordered && p.shaders.iter().any(|s| s.stage == "vertex") {
            self.requested_vertex.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else if ordered && p.graphics.as_ref().is_some_and(|g| g.writes_depth_only()) {
            self.early_depth.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    #[cfg(test)]
    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        self.pop_with(screens_only, slow_limit, true)
    }

    /// The next pipeline to make; with `early` false ([`short_of_commit`])
    /// none that no draw asked for.
    fn pop_with(&mut self, screens_only: bool, slow_limit: usize, early: bool) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested_vertex.pop_front().or_else(|| self.requested.pop_front());
            if p.is_none() && early {
                p = unclaimed(&mut self.early_depth).or_else(|| unclaimed(&mut self.early));
            }
            if p.is_none() && early && self.slow_now < slow_limit {
                p = unclaimed(&mut self.slow);
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        let early = !short_of_commit();
        if let Some(p) = queue.pop_with(screens_only, slow_limit(), early) {
            return p;
        }
        queue = if early {
            AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner())
        } else {
            // Early ones wait for the memory to come back: looked at again.
            AVAILABLE.wait_timeout(queue, Duration::from_millis(500)).unwrap_or_else(|e| e.into_inner()).0
        };
    }
}

/// Whether the PC's commit headroom is under `VICTORY_PIPELINE_COMMIT_MB`
/// (4096 by default; 0: never): only two workers make pipelines then, and
/// only those draws asked for (no early ones). A driver's compile of a
/// pipeline takes hundreds of MB of private memory for a while, and with
/// eight at once on a 32 GB PC the commit limit came near. As
/// fxceless/victory-fork PR #777 (fxceless), with its switch.
fn short_of_commit() -> bool {
    static FLOOR: OnceLock<u64> = OnceLock::new();
    let floor = *FLOOR.get_or_init(|| {
        std::env::var("VICTORY_PIPELINE_COMMIT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4096) << 20
    });
    #[cfg(windows)]
    let headroom = crate::host::commit_headroom();
    #[cfg(not(windows))]
    let headroom: Option<u64> = None;
    floor > 0 && headroom.is_some_and(|h| h < floor)
}

/// The workers that keep making pipelines while [`short_of_commit`].
const SHORT_WORKERS: usize = 2;

/// The PC's physical RAM.
fn total_ram() -> Option<u64> {
    #[cfg(windows)]
    return crate::host::total_physical_ram();
    #[cfg(not(windows))]
    None
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16, and at most 4 on
/// a PC with 32 GiB of RAM or less (`total`): each compile takes hundreds of
/// MB, as Marvisak/VIctory PR #2944 (workers by RAM). They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize, total: Option<u64>) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None if total.is_some_and(|t| t <= 32 << 30) => (processors * 2 / 3).clamp(2, 4),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
            total_ram(),
        );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    }
                    loop {
                        // Short of memory: two workers only (the screen's
                        // own are not counted).
                        while !screens_only && i >= SHORT_WORKERS && short_of_commit() {
                            std::thread::sleep(Duration::from_millis(250));
                        }
                        let p = next(screens_only);
                        p.make();
                        if p.was_slow.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.slow_now -= 1;
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        const GIB: u64 = 1 << 30;
        assert_eq!(worker_count(None, 1, None), 2);
        assert_eq!(worker_count(None, 24, None), 16);
        assert_eq!(worker_count(None, 64, Some(64 * GIB)), 16);
        assert_eq!(worker_count(Some(0), 24, None), 1);
        assert_eq!(worker_count(Some(20), 24, Some(16 * GIB)), 20);
        // At most 4 with 32 GiB of RAM or less, unless asked.
        assert_eq!(worker_count(None, 12, Some(32 * GIB - 1)), 4);
        assert_eq!(worker_count(None, 4, Some(16 * GIB)), 2);
        assert_eq!(worker_count(None, 12, Some(48 * GIB)), 8);
        assert_eq!(worker_count(Some(8), 12, Some(32 * GIB)), 8);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    /// A pipeline of one shader whose container is `text` (with the stage
    /// in it: the shader table keeps one shader per content).
    fn of_container(stage: &'static str, text: &[u8]) -> Vec<Arc<Shader>> {
        vec![super::super::shader(stage, &[stage.as_bytes(), b":", text].concat())]
    }

    #[test]
    fn ray_passes_are_left_out_only_when_switched_off() {
        let resolve = of_container("compute", b"..CS_RaytracingResolve_0db46314_Wrapped..");
        let activate = of_container("compute", b"CS_ScreenProbes_ActivateRays_1");
        let other = of_container("compute", b"CS_PedStandardSkinning_0");
        let pixel = of_container("pixel", b"CS_RaytracingResolve in a pixel shader");
        assert!(holds(b"aCCS_x", b"CS_x") && holds(b"CS_x", b"CS_x") && !holds(b"CS_", b"CS_x") && !holds(b"", b"C"));
        assert!(left_out(&resolve, [false, false]).is_none());
        assert!(left_out(&resolve, [true, false]).is_some());
        assert!(left_out(&resolve, [false, true]).is_none());
        assert!(left_out(&activate, [false, true]).is_some());
        assert!(left_out(&activate, [true, false]).is_none());
        for all in [[true, true], [false, false]] {
            assert!(left_out(&other, all).is_none());
            assert!(left_out(&pixel, all).is_none());
        }
        // One left out is never queued, made or waited for.
        let mut p = Pending::new(Vec::new(), None);
        p.left_out = Some("test");
        let p = Arc::new(p);
        early(&p);
        assert!(ready(&p).is_none());
        assert!(!now(&p));
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert!(p.made.get().is_none());
    }

    #[test]
    fn the_blend_shape_mapping_is_a_pipeline_that_fills_in_once() {
        if std::env::var_os("VICTORY_PIPELINE_ONCE").is_some_and(|v| v == "0") {
            return; // switched off by the environment
        }
        let once = |stage, text: &[u8]| Pending::new(of_container(stage, text), None).once;
        assert!(once("compute", b"CS_ComputeBlendshapeMappings_54139280_Wrapped"));
        assert!(!once("pixel", b"CS_ComputeBlendshapeMappings_54139280_Wrapped"));
        assert!(!once("compute", b"CS_PedStandardSkinning_0"));
    }

    /// The first dispatch of such a pipeline waits for the worker that is
    /// making it past the frame's budget.
    #[test]
    fn a_pipeline_that_fills_in_once_is_waited_for_past_the_budget() {
        let budget = wait_budget();
        if budget > Duration::from_secs(2) {
            return; // long by the environment
        }
        let mut p = Pending::new(Vec::new(), None);
        p.once = true;
        // A worker has taken it and is still in the driver.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);
        let entered = std::sync::Barrier::new(2);
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let (returned, result) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let (worker, entered) = (p.clone(), &entered);
            scope.spawn(move || {
                worker.made.get_or_init(|| {
                    entered.wait();
                    gate.recv().unwrap();
                    None
                });
            });
            entered.wait();
            scope.spawn(|| returned.send(ready(&p).is_some()).unwrap());
            let early = result.recv_timeout(budget + Duration::from_millis(150));
            release.send(()).unwrap();
            assert!(matches!(early, Err(std::sync::mpsc::RecvTimeoutError::Timeout)));
            assert!(!result.recv_timeout(Duration::from_secs(10)).unwrap());
        });
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// Requested pipelines with a vertex shader go before the other
    /// requested ones, early ones that write depth alone before the other
    /// early ones, and a request before both.
    #[test]
    fn the_queues_are_ordered_by_what_a_pipeline_draws() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;
        let depth_only = || {
            let mut state = Box::new(GraphicsState::default());
            let format = [16u32, DXGI_FORMAT_D32_FLOAT.0 as u32];
            // SAFETY: a whole depth format subobject of a pipeline stream.
            unsafe { state.read(16, format.as_ptr().cast()) };
            assert!(state.writes_depth_only());
            Arc::new(Pending::new(Vec::new(), Some(state)))
        };
        let of = |stage, text: &[u8]| Arc::new(Pending::new(of_container(stage, text), None));
        let (plain, promoted, shadow) = (Arc::new(Pending::new(Vec::new(), None)), depth_only(), depth_only());
        let (compute, mesh, vertex) = (of("compute", b"order"), of("mesh", b"order"), of("vertex", b"order"));
        let mut queue = Queue::new();
        for p in [&plain, &shadow, &promoted] {
            assert_eq!(queue.push_in(p, EARLY, false, true), Some(true));
        }
        for p in [&compute, &mesh, &vertex] {
            assert_eq!(queue.push_in(p, REQUESTED, false, true), Some(true));
        }
        assert_eq!(queue.push_in(&promoted, REQUESTED, false, true), Some(false));
        for expected in [&vertex, &compute, &mesh, &promoted, &shadow, &plain] {
            assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), expected));
        }
        assert!(queue.pop(false, 1).is_none());
        // Switched off: in the order asked for.
        let (compute, vertex) = (of("compute", b"order 2"), of("vertex", b"order 2"));
        let (plain, shadow) = (Arc::new(Pending::new(Vec::new(), None)), depth_only());
        for (p, priority) in [(&plain, EARLY), (&shadow, EARLY), (&compute, REQUESTED), (&vertex, REQUESTED)] {
            assert_eq!(queue.push_in(p, priority, false, false), Some(true));
        }
        for expected in [&compute, &vertex, &plain, &shadow] {
            assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), expected));
        }
    }

    #[test]
    fn a_pipeline_is_waited_for_one_budget_in_all() {
        let ms = Duration::from_millis;
        // Within a frame: what the frame has left.
        assert_eq!(allowance(ms(500), ms(0), ms(0), true), ms(500));
        assert_eq!(allowance(ms(500), ms(200), ms(0), true), ms(300));
        assert_eq!(allowance(ms(500), ms(600), ms(0), true), ms(0));
        // In a later frame: what the pipeline has left of its one budget.
        assert_eq!(allowance(ms(500), ms(0), ms(450), true), ms(50));
        assert_eq!(allowance(ms(500), ms(100), ms(500), true), ms(0));
        // Without the cap, a budget in every frame.
        assert_eq!(allowance(ms(500), ms(100), ms(5000), false), ms(400));
    }

    /// A pipeline no worker finishes takes its budget of one frame and none
    /// of the next.
    #[test]
    fn later_frames_do_not_wait_for_a_pipeline_that_took_its_budget() {
        let budget = wait_budget();
        if budget.is_zero() || budget > Duration::from_secs(2) || !total_capped() {
            return; // switched off or long by the environment
        }
        let p = Pending::new(Vec::new(), None);
        p.queue_state.store(MAKING, Ordering::Relaxed);
        // Frames no other test counts in.
        let frame = 0x7e57_0000_0000u64;
        wait(&p, frame);
        assert!(p.made.get().is_none());
        assert_eq!(Duration::from_micros(p.waited_us.load(Ordering::Relaxed)), budget);
        let again = Instant::now();
        wait(&p, frame + 1);
        assert!(again.elapsed() < budget / 4, "waited {:?} in a later frame", again.elapsed());
    }

    #[test]
    fn early_pipelines_wait_while_short_of_memory() {
        let early = Arc::new(Pending::new(Vec::new(), None));
        let asked = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();
        assert_eq!(queue.push_as(&early, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&asked, REQUESTED, false), Some(true));
        assert!(Arc::ptr_eq(&queue.pop_with(false, 1, false).unwrap(), &asked));
        assert!(queue.pop_with(false, 1, false).is_none());
        assert!(Arc::ptr_eq(&queue.pop_with(false, 1, true).unwrap(), &early));
    }

    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }
}
