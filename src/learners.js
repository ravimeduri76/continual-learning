/* ------------------------------------------------------------------
   Continual-learning engine.
   Four learners, identical frozen embeddings, one dial different each.
   No DOM, no globals — testable under node.
   ------------------------------------------------------------------ */

const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const zeros = (d) => new Float64Array(d);

/* Deterministic PRNG so a replay of the same answers reproduces the same run. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ==================================================================
   1. ONLINE LOGISTIC, PLAIN SGD
   Locus: everything · Pace: constant η · Anchor: none
   Geometry: I · Evidence: newest example only
   The control. Nothing holds it back, so nothing holds it still.
   ================================================================== */
class SGDLogistic {
  constructor(d, { eta = 0.90, steps = 4 } = {}) {
    this.id = 'sgd'; this.d = d; this.eta = eta; this.steps = steps;
    this.w = zeros(d); this.b = 0; this.moved = 0;
  }
  score(x) { return dot(this.w, x) + this.b; }
  prob(x) { return sigmoid(this.score(x)); }
  observe(x, y) {
    const before = this.w.slice();
    for (let s = 0; s < this.steps; s++) {
      const g = sigmoid(dot(this.w, x) + this.b) - y;
      for (let i = 0; i < this.d; i++) this.w[i] -= this.eta * g * x[i];
      this.b -= this.eta * g;
    }
    this.moved = l2diff(before, this.w);
  }
  /* Effective memory: a constant step size forgets geometrically. */
  nEff() { return 1 / this.eta; }
}

/* ==================================================================
   2. FTRL-PROXIMAL  (McMahan et al., KDD 2013)
   Same loss as (1). One dial changed: it is anchored to its own
   previous iterates, with per-coordinate stiffness sqrt(sum g^2).
   That stiffness is the same statistic EWC calls the Fisher.
   ================================================================== */
class FTRLProximal {
  constructor(d, { alpha = 1.20, beta = 1.0, l1 = 0.004, l2 = 1.0, steps = 4 } = {}) {
    this.id = 'ftrl'; this.d = d;
    this.alpha = alpha; this.beta = beta; this.l1 = l1; this.l2 = l2; this.steps = steps;
    this.z = zeros(d); this.n = zeros(d); this.w = zeros(d);
    this.zb = 0; this.nb = 0; this.b = 0; this.moved = 0;
  }
  _materialise() {
    for (let i = 0; i < this.d; i++) {
      const z = this.z[i];
      if (Math.abs(z) <= this.l1) { this.w[i] = 0; }
      else {
        const sgn = z < 0 ? -1 : 1;
        this.w[i] = -(z - sgn * this.l1) /
                    ((this.beta + Math.sqrt(this.n[i])) / this.alpha + this.l2);
      }
    }
    this.b = -this.zb / ((this.beta + Math.sqrt(this.nb)) / this.alpha + this.l2);
  }
  score(x) { this._materialise(); return dot(this.w, x) + this.b; }
  prob(x) { return sigmoid(this.score(x)); }
  observe(x, y) {
    const before = this.w.slice();
    for (let s = 0; s < this.steps; s++) {
      this._materialise();
      const p = sigmoid(dot(this.w, x) + this.b);
      const err = p - y;
      for (let i = 0; i < this.d; i++) {
        const g = err * x[i];
        // sigma_i = (1/alpha)(sqrt(n+g^2) - sqrt(n))  ==  1/eta_t - 1/eta_{t-1}
        const sigma = (Math.sqrt(this.n[i] + g * g) - Math.sqrt(this.n[i])) / this.alpha;
        this.z[i] += g - sigma * this.w[i];   // <- the proximal correction
        this.n[i] += g * g;
      }
      const gb = err;
      const sb = (Math.sqrt(this.nb + gb * gb) - Math.sqrt(this.nb)) / this.alpha;
      this.zb += gb - sb * this.b;
      this.nb += gb * gb;
    }
    this._materialise();
    this.moved = l2diff(before, this.w);
  }
  /* Stiffness grows as sqrt(sum g^2); the implied window is its inverse. */
  nEff() {
    let m = 0; for (let i = 0; i < this.d; i++) m += this.n[i];
    return 1 + Math.sqrt(m / this.d) / this.alpha;
  }
  sparsity() {
    let z = 0; for (let i = 0; i < this.d; i++) if (this.w[i] === 0) z++;
    return z / this.d;
  }
}

/* ==================================================================
   3. PROTOTYPE / NEAREST-CENTROID
   No gradients at all. Closed form, one-shot, order-invariant.
   gamma < 1 turns it into an exponentially-forgetting centroid.
   This is, structurally, what in-context learning does.
   ================================================================== */
class Prototype {
  constructor(d, { gamma = 1.0, tau = 6.0 } = {}) {
    this.id = 'proto'; this.d = d; this.gamma = gamma; this.tau = tau;
    this.mp = zeros(d); this.mn = zeros(d);
    this.np = 0; this.nn = 0; this.moved = 0;
  }
  score(x) {
    const a = this.np > 0 ? cosine(x, this.mp) : 0;
    const b = this.nn > 0 ? cosine(x, this.mn) : 0;
    return this.tau * (a - b);
  }
  prob(x) { return sigmoid(this.score(x)); }
  observe(x, y) {
    const tgt = y === 1 ? this.mp : this.mn;
    const before = tgt.slice();
    const g = this.gamma;
    for (let i = 0; i < this.d; i++) tgt[i] = g * tgt[i] + x[i];
    if (y === 1) this.np = g * this.np + 1; else this.nn = g * this.nn + 1;
    this.moved = l2diff(norm(before), norm(tgt));
  }
  /* With gamma=1 it answers to every example it has ever seen. */
  nEff() { return this.gamma >= 1 ? (this.np + this.nn) : 1 / (1 - this.gamma); }
}

/* ==================================================================
   4. REPLAY + FISHER-WEIGHTED DRIFT PENALTY  (online-EWC shape)
   Locus: everything · Anchor: its own previous solution
   Geometry: diagonal empirical Fisher · Evidence: stream + full replay
   Same skeleton as FTRL. That is the point.
   ================================================================== */
class ReplayEWC {
  constructor(d, { eta = 0.70, lambda = 0.12, steps = 8, gammaF = 0.92, seed = 7 } = {}) {
    this.id = 'replay'; this.d = d;
    this.eta = eta; this.lambda = lambda; this.steps = steps; this.gammaF = gammaF;
    this.w = zeros(d); this.b = 0;
    this.F = zeros(d); this.anchor = zeros(d);
    this.buf = []; this.moved = 0; this.rng = mulberry32(seed);
  }
  score(x) { return dot(this.w, x) + this.b; }
  prob(x) { return sigmoid(this.score(x)); }
  observe(x, y) {
    const before = this.w.slice();
    this.buf.push({ x, y });
    // Anchor to the previous solution, not to the origin (Law II).
    this.anchor = this.w.slice();

    const B = Math.min(this.buf.length, 12);
    for (let s = 0; s < this.steps; s++) {
      const grad = zeros(this.d); let gb = 0;
      const batch = [];
      batch.push(this.buf[this.buf.length - 1]);              // always the newest
      while (batch.length < B) {                               // plus rehearsal
        batch.push(this.buf[Math.floor(this.rng() * this.buf.length)]);
      }
      for (const ex of batch) {
        const err = sigmoid(dot(this.w, ex.x) + this.b) - ex.y;
        for (let i = 0; i < this.d; i++) grad[i] += err * ex.x[i];
        gb += err;
      }
      for (let i = 0; i < this.d; i++) {
        const g = grad[i] / batch.length;
        // running diagonal empirical Fisher = decayed mean of squared gradients
        this.F[i] = this.gammaF * this.F[i] + (1 - this.gammaF) * g * g;
        const pull = this.lambda * (this.F[i] + 1e-3) * (this.w[i] - this.anchor[i]);
        this.w[i] -= this.eta * (g + pull);
      }
      this.b -= this.eta * (gb / batch.length);
    }
    this.moved = l2diff(before, this.w);
  }
  nEff() { return this.buf.length; }
}

/* ==================================================================
   5. FROZEN BACKBONE + RANK-r ADAPTER  (the LP-FT demonstration)
   Off by default. Unfreezing a small adapter alongside the head
   should IMPROVE recent items and DEGRADE the earliest ones —
   Kumar et al. (ICLR 2022), reproduced at a scale you can watch.
   ================================================================== */
class AdapterHead {
  constructor(d, { r = 2, eta = 0.90, etaAdapter = 0.30, steps = 4, seed = 11, cap = 1.0 } = {}) {
    this.id = 'adapter'; this.d = d; this.r = r;
    this.eta = eta; this.etaA = etaAdapter; this.steps = steps; this.cap = cap;
    this.w = zeros(d); this.b = 0; this.moved = 0;
    const rng = mulberry32(seed);
    this.U = []; this.V = [];
    for (let j = 0; j < r; j++) {
      const u = zeros(d), v = zeros(d);
      for (let i = 0; i < d; i++) { u[i] = (rng() - 0.5) * 0.02; v[i] = (rng() - 0.5) * 0.02; }
      this.U.push(u); this.V.push(v);
    }
  }
  _feat(x) {                       // z = x + sum_j u_j (v_j . x)
    const z = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) z[i] = x[i];
    this.proj = [];
    for (let j = 0; j < this.r; j++) {
      const c = dot(this.V[j], x); this.proj.push(c);
      for (let i = 0; i < x.length; i++) z[i] += this.U[j][i] * c;
    }
    return z;
  }
  score(x) { return dot(this.w, this._feat(x)) + this.b; }
  prob(x) { return sigmoid(this.score(x)); }
  observe(x, y) {
    const before = this.w.slice();
    for (let s = 0; s < this.steps; s++) {
      const z = this._feat(x);
      const err = sigmoid(dot(this.w, z) + this.b) - y;
      // head
      for (let i = 0; i < this.d; i++) this.w[i] -= this.eta * err * z[i];
      this.b -= this.eta * err;
      // adapter — this is what distorts the frozen features
      for (let j = 0; j < this.r; j++) {
        const c = this.proj[j];
        const wu = dot(this.w, this.U[j]);
        for (let i = 0; i < this.d; i++) {
          this.U[j][i] -= this.etaA * err * this.w[i] * c;
          this.V[j][i] -= this.etaA * err * wu * x[i];
        }
        // project each factor back into a ball: an adapter that can grow without
        // bound stops being an adapter and becomes a divergence
        clip(this.U[j], this.cap); clip(this.V[j], this.cap);
      }
    }
    this.moved = l2diff(before, this.w);
  }
  nEff() { return 1 / this.eta; }
}

/* ==================================================================
   6. TWO-CLOCK  (slow centroid body + fast leaky logistic head)
   The escape from Law I. Every learner above has ONE clock, so it can
   only slide along the retention/adaptation frontier. This one partitions
   the predictor into two tiers of DIFFERENT geometry — a genuine
   partition, not one head in disguise:

     score(x) = tau (cos(x,μ₊) − cos(x,μ₋))   ← SLOW: closed-form centroid,
              + wF·x + bF                        gamma≈1, answers to every
                                                 example, order-invariant.
                                               ← FAST: leaky logistic on the
                                                 slow tier's RESIDUAL, high η,
                                                 decays → N_eff ≈ 2.

   Why the partition is real: two linear heads summed collapse to w_s+w_f —
   one head, one clock, stuck on the frontier. A centroid score is nonlinear
   in x (normalisation, separate ± means), so this sum does NOT reduce to a
   single logistic. Slow keeps the old concept; fast chases the new one.
   The Continuum Memory System shape (arXiv:2512.24695) at watchable scale.
   It does NOT close the consolidation gap (Invariant 5): the two traces
   never merge into one store — retention lives in μ, adaptation in wF, and
   the fast trace is simply discarded as it leaks.
   ================================================================== */
class TwoClock {
  constructor(d, {
    etaF = 0.90, rhoF = 0.5,          // fast head: big step then leak -> N_eff = 2
    gammaS = 1.0, tau = 8.0,          // slow centroid: gamma=1 -> never forgets
  } = {}) {
    this.id = 'twoclock'; this.d = d;
    this.etaF = etaF; this.rhoF = rhoF; this.gammaS = gammaS; this.tau = tau;
    this.mp = zeros(d); this.mn = zeros(d); this.np = 0; this.nn = 0;   // slow
    this.wF = zeros(d); this.bF = 0;                                     // fast
    this.t = 0; this.moved = 0;
  }
  _slow(x) {
    const a = this.np > 0 ? cosine(x, this.mp) : 0;
    const b = this.nn > 0 ? cosine(x, this.mn) : 0;
    return this.tau * (a - b);
  }
  score(x) { return this._slow(x) + dot(this.wF, x) + this.bF; }
  prob(x) { return sigmoid(this.score(x)); }
  observe(x, y) {
    const beforeF = this.wF.slice();
    this.t++;
    // Error at arrival, before either tier adapts to this point.
    const errPre = sigmoid(this.score(x)) - y;
    // FAST head: descend the residual the slow centroid did not already explain,
    // then leak so it holds only recent evidence.
    for (let i = 0; i < this.d; i++) this.wF[i] = this.rhoF * (this.wF[i] - this.etaF * errPre * x[i]);
    this.bF = this.rhoF * (this.bF - this.etaF * errPre);
    // SLOW body: closed-form centroid, gamma=1 answers to everything it ever saw.
    const g = this.gammaS, tgt = y === 1 ? this.mp : this.mn;
    for (let i = 0; i < this.d; i++) tgt[i] = g * tgt[i] + x[i];
    if (y === 1) this.np = g * this.np + 1; else this.nn = g * this.nn + 1;
    this.moved = l2diff(beforeF, this.wF);
  }
  nEffFast() { return 1 / (1 - this.rhoF); }
  nEffSlow() { return this.gammaS >= 1 ? (this.np + this.nn) : 1 / (1 - this.gammaS); }
  nEff() { return this.nEffSlow(); }                        // headline = the long clock
}

/* ------------------------------- utils ------------------------------- */
function clip(v, cap) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; s = Math.sqrt(s); if (s > cap) { const f = cap / s; for (let i = 0; i < v.length; i++) v[i] *= f; } }
function l2diff(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
function norm(v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; s = Math.sqrt(s) || 1; const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] / s; return o; }
function cosine(a, b) { const na = Math.sqrt(dot(a, a)) || 1e-9, nb = Math.sqrt(dot(b, b)) || 1e-9; return dot(a, b) / (na * nb); }

/* ==================================================================
   DRIFT DETECTION
   Page-Hinkley works at the sample sizes a game can produce.
   ADWIN is also computed, honestly, to show it cannot fire yet.
   ================================================================== */
class PageHinkley {
  constructor({ delta = 0.05, threshold = 0.75 } = {}) {
    this.delta = delta; this.threshold = threshold;
    this.n = 0; this.mean = 0; this.mT = 0; this.MT = 0; this.fired = false; this.firedAt = null;
  }
  push(err) {                       // err in {0,1}: was the prediction wrong?
    this.n++;
    this.mean += (err - this.mean) / this.n;
    this.mT += err - this.mean - this.delta;
    this.MT = Math.min(this.MT, this.mT);
    const stat = this.mT - this.MT;
    if (!this.fired && this.n >= 4 && stat > this.threshold) { this.fired = true; this.firedAt = this.n; }
    return stat;
  }
}

/* ADWIN's cut threshold at the current window size — reported, not used,
   so the player can see exactly how much data a guaranteed detector needs. */
function adwinEpsCut(n0, n1, delta = 0.2) {
  if (n0 < 1 || n1 < 1) return Infinity;
  const m = 1 / (1 / n0 + 1 / n1);
  const dp = delta / (n0 + n1);
  return Math.sqrt(Math.log(4 / dp) / (2 * m));
}

/* ------------------------------------------------------------------ */
function makeLearners(d, opts = {}) {
  return [
    new SGDLogistic(d, opts.sgd),
    new FTRLProximal(d, opts.ftrl),
    new Prototype(d, opts.proto),
    new ReplayEWC(d, opts.replay),
  ];
}

const LEARNER_META = {
  sgd:     { name: 'Online logistic (SGD)', short: 'SGD',       dial: 'the control — no anchor, no replay' },
  ftrl:    { name: 'FTRL-Proximal',         short: 'FTRL',      dial: 'Anchor + Geometry changed' },
  proto:   { name: 'Prototype centroid',    short: 'Prototype', dial: 'Pace = one shot, closed form' },
  replay:  { name: 'Replay + Fisher anchor',short: 'Replay',    dial: 'Evidence + Anchor + Geometry' },
  adapter: { name: 'Head + rank-2 adapter', short: 'Adapter',   dial: 'Locus widened — watch it distort' },
  twoclock:{ name: 'Two-clock (fast+slow)',  short: 'TwoClock',  dial: 'Partition — two clocks, escapes Law I' },
};

if (typeof module !== 'undefined') {
  module.exports = { SGDLogistic, FTRLProximal, Prototype, ReplayEWC, AdapterHead, TwoClock,
                     PageHinkley, adwinEpsCut, makeLearners, LEARNER_META, sigmoid, dot, mulberry32 };
}
