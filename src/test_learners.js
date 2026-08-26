const L = require('./learners.js');

function rngN(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function unit(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map(x => x / s); }

const D = Number(process.argv[2] || 24);
const K = 8;                       // latent clusters, like real item categories
const rng = L.mulberry32(4242);

const centres = [];
for (let k = 0; k < K; k++) centres.push(unit(Array.from({ length: D }, () => rngN(rng))));

const ITEMS = [];
for (let i = 0; i < 600; i++) {
  const k = i % K;
  const s = 0.6 / Math.sqrt(D);
  const x = centres[k].map(c => c + rngN(rng) * s);
  ITEMS.push({ x: Float64Array.from(unit(x)), k });
}
for (let i = ITEMS.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [ITEMS[i], ITEMS[j]] = [ITEMS[j], ITEMS[i]]; }

// Concept = "I like items from this set of clusters".
const CONCEPT_A = new Set([0, 1, 2, 3]);
const CONCEPT_B = new Set([2, 3, 4, 5]);   // overlapping -> genuine conflict on 0,1 and 4,5
const lab = (it, C) => (C.has(it.k) ? 1 : 0);

const NAMES = ['sgd', 'ftrl', 'proto', 'replay'];
const pct = v => (v * 100).toFixed(1).padStart(6) + '%';
const line = (lbl, o) => console.log(lbl.padEnd(32) + NAMES.map(k => pct(o[k])).join(' '));
const evalOn = (ls, data) => Object.fromEntries(ls.map(m => [m.id,
  data.filter(({ it, y }) => ((m.prob(it.x) >= .5 ? 1 : 0) === y)).length / data.length]));

function balancedDraw(pool, C, n, cursor) {
  const out = [];
  let want = 1;
  while (out.length < n && cursor.i < pool.length) {
    const it = pool[cursor.i++];
    const y = lab(it, C);
    if (y === want || out.length >= n - 1) { out.push({ it, y }); want = 1 - want; }
  }
  return out;
}

function run(drift, opts) {
  const ls = L.makeLearners(D, opts);
  const ph = new L.PageHinkley();
  const cur = { i: 0 };
  const C2 = drift ? CONCEPT_B : CONCEPT_A;

  const act1 = balancedDraw(ITEMS, CONCEPT_A, 9, cur);
  const test1 = balancedDraw(ITEMS, CONCEPT_A, 40, cur);      // held out, never trained on
  const act2 = balancedDraw(ITEMS, C2, 7, cur);
  const test2 = balancedDraw(ITEMS, C2, 40, cur);

  for (const { it, y } of act1) {
    for (const m of ls) m.observe(it.x, y);
    ph.push(Math.abs(Math.round(ls[3].prob(it.x)) - y));
  }
  const mid = evalOn(ls, test1);

  for (const { it, y } of act2) {
    const pre = ls[3].prob(it.x);
    for (const m of ls) m.observe(it.x, y);
    ph.push(Math.abs(Math.round(pre) - y));
  }
  const oldAfter = evalOn(ls, test1);
  const newAfter = evalOn(ls, test2);
  return { ls, mid, oldAfter, newAfter, ph };
}

console.log(`\nd=${D}, ${K} latent clusters, 9 train -> 40 held-out -> 7 train -> 40 held-out`);
console.log(' '.repeat(32) + '   SGD     FTRL   PROTO  REPLAY');

console.log('\n--- STATIONARY (no drift) ---');
let r = run(false);
line('held-out after 9', r.mid);
line('held-out after 16', r.newAfter);

console.log('\n--- DRIFT at example 10 ---');
r = run(true);
line('OLD concept, before drift', r.mid);
line('OLD concept, after drift', r.oldAfter);
line('NEW concept, after drift', r.newAfter);
const bwt = Object.fromEntries(NAMES.map(k => [k, r.oldAfter[k] - r.mid[k]]));
line('backward transfer (BWT)', bwt);
console.log('\nPage-Hinkley:', r.ph.fired ? `fired at example ${r.ph.firedAt}` : 'did not fire');
console.log('ADWIN eps_cut(9,7) =', L.adwinEpsCut(9, 7).toFixed(3), '- unreachable at this n');

console.log('\n--- dial readouts ---');
for (const m of r.ls) console.log(`  ${m.id.padEnd(8)} N_eff=${m.nEff().toFixed(1).padStart(6)}  last |Δw|=${m.moved.toFixed(3)}`);

console.log('\n--- LP-FT: unfrozen rank-2 adapter vs frozen backbone ---');
{
  const cur = { i: 300 };
  const tr = balancedDraw(ITEMS, CONCEPT_A, 16, cur);
  const early = tr.slice(0, 6), late = tr.slice(-6);
  const ho = balancedDraw(ITEMS, CONCEPT_A, 60, cur);
  const head = new L.SGDLogistic(D), ad = new L.AdapterHead(D);
  const acc = (m, s) => s.filter(({ it, y }) => (m.prob(it.x) >= .5 ? 1 : 0) === y).length / s.length;
  for (const { it, y } of tr) { head.observe(it.x, y); ad.observe(it.x, y); }
  console.log(`  frozen backbone + head   earliest-6=${pct(acc(head, early))} latest-6=${pct(acc(head, late))} held-out=${pct(acc(head, ho))}`);
  console.log(`  + unfrozen rank-2 adapter earliest-6=${pct(acc(ad, early))} latest-6=${pct(acc(ad, late))} held-out=${pct(acc(ad, ho))}`);
}
console.log('');
