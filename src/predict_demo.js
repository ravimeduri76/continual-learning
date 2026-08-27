/* ------------------------------------------------------------------
   PREDICT capability — can 5 answers name a liked 6th and an unliked 7th?

   The original brief: ask 5-10 like/pass questions, then predict the next
   liked / unliked items. This measures exactly that on the real CLIP bundle.

   Mirrors the game: farthest-point-sample N_SHOWN items (default 5), label
   each by a real ground-truth concept, fit all five learners, then rank the
   REMAINING pool by P(you say yes) and read off
     top-1    = the predicted 6th item (should be LIKED)
     bottom-1 = the predicted 7th item (should be UNLIKED)
   and check both against ground truth, over many draws and four concepts
   per mode.

   The 6th-liked number is the hard one (base like-rate ~30%); the 7th-unliked
   is easier because most items genuinely are not liked. Both are reported
   against the random-pick baseline so the lift is legible.

   Run:  node predict_demo.js [trials=400] [nShown=5]
   ------------------------------------------------------------------ */
const fs = require('fs');
const CL = require('./learners.js');
const B = JSON.parse(fs.readFileSync(require('path').join(__dirname, 'assets/bundle.json'), 'utf8'));

/* same int8 -> unit-vector decode the game uses */
function decode(mode) {
  const m = B[mode], bin = Buffer.from(m.vecs, 'base64');
  const q = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) q[i] = (bin[i] << 24) >> 24;
  const V = [];
  for (let i = 0; i < m.items.length; i++) {
    const v = new Float64Array(m.dim); let s = 0;
    for (let j = 0; j < m.dim; j++) { v[j] = q[i * m.dim + j] / 127 * m.scale; s += v[j] * v[j]; }
    s = Math.sqrt(s) || 1;
    for (let j = 0; j < m.dim; j++) v[j] /= s;
    V.push(v);
  }
  return { V, items: m.items, d: m.dim };
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/* farthest-point sampling of `n` indices from `start` — herding, as the game does */
function fps(V, n, start) {
  const N = V.length, chosen = [start], best = new Float64Array(N).fill(Infinity);
  for (let k = 1; k < n; k++) {
    const last = V[chosen[chosen.length - 1]];
    let arg = -1, argv = -Infinity;
    for (let i = 0; i < N; i++) {
      if (chosen.includes(i)) continue;
      const dist = 1 - dot(last, V[i]);
      if (dist < best[i]) best[i] = dist;
      if (best[i] > argv) { argv = best[i]; arg = i; }
    }
    chosen.push(arg);
  }
  return chosen;
}

/* ground-truth concepts drawn from the items' own attributes */
const CONCEPTS = {
  shapes: {
    'stars & crosses': it => ['star5', 'star6', 'cross'].includes(it.family),
    'round / curved':  it => it.curved === true,
    'solid fill':      it => it.fill === 'solid',
    'pointy':          it => it.pointy === true,
  },
  flags: {
    'has green':        it => it.colours.includes('green'),
    'European':         it => it.continent === 'Europe',
    'horizontal bands': it => it.bands === 'horizontal',
    'African':          it => it.continent === 'Africa',
  },
  people: {
    'sport':          it => it.field === 'sport',
    'science & tech': it => it.field === 'science' || it.field === 'tech',
    'historical':     it => it.era === 'historical',
    'from Europe':    it => it.region === 'Europe',
  },
};

const LEARNERS = ['sgd', 'ftrl', 'proto', 'replay', 'twoclock'];
function makeAll(d) { const ls = CL.makeLearners(d, {}); ls.push(new CL.TwoClock(d)); return ls; }

const TRIALS = Number(process.argv[2] || 400);
const N_SHOWN = Number(process.argv[3] || 5);

function run(mode) {
  const { V, items, d } = decode(mode);
  const N = V.length;
  const rng = CL.mulberry32(20260827 + mode.length);
  const acc = {}; for (const L of LEARNERS) acc[L] = { like: 0, unlike: 0, both: 0, n: 0 };
  let baseSum = 0, nConcepts = 0, informative = 0, draws = 0;

  for (const [, fn] of Object.entries(CONCEPTS[mode])) {
    const truth = items.map(fn);
    baseSum += truth.filter(Boolean).length / N; nConcepts++;

    for (let t = 0; t < TRIALS; t++) {
      const shown = fps(V, N_SHOWN, Math.floor(rng() * N));
      const ys = shown.map(i => (truth[i] ? 1 : 0));
      draws++;
      if (ys.includes(1) && ys.includes(0)) informative++;   // the 5 had signal on both sides

      const ls = makeAll(d);
      for (let s = 0; s < shown.length; s++) for (const m of ls) m.observe(V[shown[s]], ys[s]);

      const shownSet = new Set(shown), pool = [];
      for (let i = 0; i < N; i++) if (!shownSet.has(i)) pool.push(i);

      for (const m of ls) {
        let top = -1, topP = -Infinity, bot = -1, botP = Infinity;
        for (const i of pool) {
          const p = m.prob(V[i]);
          if (p > topP) { topP = p; top = i; }
          if (p < botP) { botP = p; bot = i; }
        }
        const likeOK = truth[top] === true, unlikeOK = truth[bot] === false;
        const a = acc[m.id];
        a.like += likeOK; a.unlike += unlikeOK; a.both += (likeOK && unlikeOK); a.n++;
      }
    }
  }
  return { acc, base: baseSum / nConcepts, informative: informative / draws };
}

const pct = v => (v * 100).toFixed(0) + '%';
console.log(`\nPredict the 6th (liked) and 7th (unliked) item from ${N_SHOWN} answers`);
console.log(`${TRIALS} draws x 4 concepts per mode, real CLIP embeddings, top-1 / bottom-1 of the ranked pool\n`);
for (const mode of ['shapes', 'flags', 'people']) {
  const r = run(mode);
  console.log(`-- ${mode.toUpperCase()}  (avg concept base rate ${pct(r.base)}; ${pct(r.informative)} of ${N_SHOWN}-draws saw a like AND a dislike)`);
  console.log('   learner      6th liked    7th unliked   both');
  for (const L of LEARNERS) {
    const a = r.acc[L];
    console.log('   ' + L.padEnd(11) + pct(a.like / a.n).padStart(8) + pct(a.unlike / a.n).padStart(14) + pct(a.both / a.n).padStart(9));
  }
  console.log(`   random pick: ${pct(r.base)} liked, ${pct(1 - r.base)} unliked, ${pct(r.base * (1 - r.base))} both\n`);
}
