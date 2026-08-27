/* Validate the learners on the REAL CLIP embeddings with simulated players. */
const fs = require('fs');
const L = require('./learners.js');

const B = JSON.parse(fs.readFileSync('assets/bundle.json'));

function decode(mode) {
  const m = B[mode];
  const buf = Buffer.from(m.vecs, 'base64');
  const q = new Int8Array(buf.buffer, buf.byteOffset, buf.length);
  const n = m.items.length, d = m.dim;
  const V = [];
  for (let i = 0; i < n; i++) {
    const v = new Float64Array(d);
    let s = 0;
    for (let j = 0; j < d; j++) { v[j] = q[i * d + j] / 127 * m.scale; s += v[j] * v[j]; }
    s = Math.sqrt(s) || 1;
    for (let j = 0; j < d; j++) v[j] /= s;
    V.push(v);
  }
  return { V, items: m.items, d };
}

/* Simulated players: a rule over the KNOWN generative factors. */
const CONCEPTS = {
  shapes: {
    A: { label: 'round things',        f: i => ['circle', 'ring', 'crescent'].includes(i.family) },
    B: { label: 'warm colours',        f: i => ['red', 'orange', 'amber', 'pink', 'magenta'].includes(i.colour) },
  },
  flags: {
    A: { label: 'has green in it',     f: i => i.colours.includes('green') },
    B: { label: 'European flags',      f: i => i.continent === 'Europe' },
  },
  people: {
    A: { label: 'tech leaders',        f: i => i.industry === 'tech' },
    B: { label: 'from Asia',           f: i => i.region === 'Asia' },
  },
};

function rngFactory(seed) { return L.mulberry32(seed); }

function draw(pool, concept, n, used, rng) {
  /* alternate liked / not-liked so the stream is balanced, as the game will do */
  const out = []; let want = true; let guard = 0;
  while (out.length < n && guard++ < 20000) {
    const i = Math.floor(rng() * pool.length);
    if (used.has(i)) continue;
    const y = concept.f(pool[i]) ? 1 : 0;
    if ((y === 1) === want) { used.add(i); out.push({ i, y }); want = !want; }
  }
  return out;
}

const NAMES = ['sgd', 'ftrl', 'proto', 'replay'];
const pct = v => (v * 100).toFixed(0).padStart(5) + '%';

function protocol(mode, opts, seed) {
  const { V, items, d } = decode(mode);
  const C = CONCEPTS[mode];
  const rng = rngFactory(seed);
  const used = new Set();
  const ls = L.makeLearners(d, opts);
  const ph = new L.PageHinkley();

  const act1 = draw(items, C.A, 9, used, rng);
  const test1 = draw(items, C.A, 40, used, rng);
  const act2 = draw(items, C.B, 8, used, rng);
  const test2 = draw(items, C.B, 40, used, rng);

  const ev = set => Object.fromEntries(ls.map(m => [m.id,
    set.filter(({ i, y }) => ((m.prob(V[i]) >= .5 ? 1 : 0) === y)).length / set.length]));

  for (const { i, y } of act1) { for (const m of ls) m.observe(V[i], y); }
  const mid = ev(test1);
  for (const { i, y } of act2) {
    ph.push(Math.abs(Math.round(ls[3].prob(V[i])) - y));
    for (const m of ls) m.observe(V[i], y);
  }
  return { mid, oldAfter: ev(test1), newAfter: ev(test2), ph, ls };
}

function avg(mode, opts, runs = 24) {
  const acc = { mid: {}, oldAfter: {}, newAfter: {} };
  let fired = 0;
  for (const k of NAMES) for (const p of ['mid', 'oldAfter', 'newAfter']) acc[p][k] = 0;
  for (let s = 0; s < runs; s++) {
    const r = protocol(mode, opts, 1000 + s * 17);
    for (const p of ['mid', 'oldAfter', 'newAfter'])
      for (const k of NAMES) acc[p][k] += r[p][k] / runs;
    if (r.ph.fired) fired++;
  }
  acc.fired = fired / runs;
  return acc;
}

const opts = {};
console.log('\nAveraged over 24 simulated players.  9 train -> 40 held-out -> DRIFT -> 8 train -> 40 held-out\n');
for (const mode of ['shapes', 'flags', 'people']) {
  const C = CONCEPTS[mode];
  const a = avg(mode, opts);
  console.log(`${mode.toUpperCase()}   "${C.A.label}"  ->  "${C.B.label}"`);
  console.log('  ' + ' '.repeat(28) + '  SGD   FTRL  PROTO REPLAY');
  const line = (l, o) => console.log('  ' + l.padEnd(28) + NAMES.map(k => pct(o[k])).join(' '));
  line('old concept, before drift', a.mid);
  line('old concept, after drift', a.oldAfter);
  line('new concept, after drift', a.newAfter);
  line('backward transfer', Object.fromEntries(NAMES.map(k => [k, a.oldAfter[k] - a.mid[k]])));
  console.log(`  drift detected in ${(a.fired * 100).toFixed(0)}% of runs\n`);
}
