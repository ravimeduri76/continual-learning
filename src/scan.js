/* Which preference concepts are actually learnable from ~9 examples? */
const fs = require('fs');
const L = require('./learners.js');
const B = JSON.parse(fs.readFileSync('assets/bundle.json'));

function decode(mode) {
  const m = B[mode];
  const buf = Buffer.from(m.vecs, 'base64');
  const q = new Int8Array(buf.buffer, buf.byteOffset, buf.length);
  const V = [];
  for (let i = 0; i < m.items.length; i++) {
    const v = new Float64Array(m.dim); let s = 0;
    for (let j = 0; j < m.dim; j++) { v[j] = q[i * m.dim + j] / 127 * m.scale; s += v[j] * v[j]; }
    s = Math.sqrt(s) || 1; for (let j = 0; j < m.dim; j++) v[j] /= s;
    V.push(v);
  }
  return { V, items: m.items, d: m.dim };
}

const IN = (k, ...vals) => i => vals.includes(i[k]);
const CANDIDATES = {
  shapes: {
    'round things':        IN('family', 'circle', 'ring', 'crescent'),
    'sharp / pointy':      i => i.pointy,
    'outlines not fills':  IN('fill', 'outline'),
    'patterned fills':     IN('fill', 'dots', 'stripes'),
    'cool colours':        IN('colour', 'blue', 'teal', 'cyan', 'indigo', 'green'),
    'warm colours':        IN('colour', 'red', 'orange', 'amber', 'pink', 'magenta'),
    'monochrome':          i => i.chroma === 'achromatic',
    'groups of three':     i => i.count === 3,
    'stars & crosses':     IN('family', 'star5', 'star6', 'cross'),
    'polygons':            IN('family', 'square', 'pentagon', 'hexagon', 'diamond', 'triangle'),
  },
  flags: {
    'horizontal bands':    IN('bands', 'horizontal'),
    'vertical bands':      IN('bands', 'vertical'),
    'mostly red':          i => i.colours[0] === 'red',
    'mostly blue':         i => i.colours[0] === 'blue',
    'mostly green':        i => i.colours[0] === 'green',
    'has green in it':     i => i.colours.includes('green'),
    'Africa':              IN('continent', 'Africa'),
    'Europe':              IN('continent', 'Europe'),
    'Asia':                IN('continent', 'Asia'),
    'Africa or Asia':      IN('continent', 'Africa', 'Asia'),
    'the Americas':        IN('continent', 'North America', 'South America'),
  },
  people: {
    'tech leaders':        IN('industry', 'tech'),
    'finance':             IN('industry', 'finance'),
    'crypto or finance':   IN('industry', 'crypto', 'finance'),
    'consumer brands':     IN('industry', 'retail', 'media'),
    'from Asia':           IN('region', 'Asia'),
    'from the Americas':   IN('region', 'Americas'),
    'from Europe':         IN('region', 'Europe'),
  },
};

function trial(V, items, fn, nTrain, seed, opts) {
  const rng = L.mulberry32(seed);
  const used = new Set();
  const pick = (n) => {
    const out = []; let want = true; let g = 0;
    while (out.length < n && g++ < 40000) {
      const i = Math.floor(rng() * items.length);
      if (used.has(i)) continue;
      const y = fn(items[i]) ? 1 : 0;
      if ((y === 1) === want) { used.add(i); out.push({ i, y }); want = !want; }
    }
    return out;
  };
  const tr = pick(nTrain);
  if (tr.length < nTrain) return null;
  const te = pick(50);
  const ls = L.makeLearners(V[0].length, opts);
  for (const { i, y } of tr) for (const m of ls) m.observe(V[i], y);
  return Object.fromEntries(ls.map(m => [m.id,
    te.filter(({ i, y }) => ((m.prob(V[i]) >= .5 ? 1 : 0) === y)).length / te.length]));
}

const NAMES = ['sgd', 'ftrl', 'proto', 'replay'];
const opts = JSON.parse(process.env.OPTS || '{}');
const NTRAIN = Number(process.env.NTRAIN || 9);
const RUNS = 30;

for (const mode of ['shapes', 'flags', 'people']) {
  const { V, items } = decode(mode);
  const rows = [];
  for (const [label, fn] of Object.entries(CANDIDATES[mode])) {
    const share = items.filter(fn).length / items.length;
    const acc = Object.fromEntries(NAMES.map(k => [k, 0]));
    let ok = 0;
    for (let s = 0; s < RUNS; s++) {
      const r = trial(V, items, fn, NTRAIN, 900 + s * 31, opts);
      if (!r) continue;
      ok++; for (const k of NAMES) acc[k] += r[k];
    }
    if (!ok) { rows.push([label, share, null]); continue; }
    for (const k of NAMES) acc[k] /= ok;
    rows.push([label, share, acc, (acc.proto + acc.replay + acc.ftrl) / 3]);
  }
  rows.sort((a, b) => (b[3] || 0) - (a[3] || 0));
  console.log(`\n${mode.toUpperCase()}  (n=${NTRAIN} train, 50 held-out, ${RUNS} seeds)`);
  console.log('  concept                 share    SGD  FTRL PROTO REPLY');
  for (const [label, share, acc] of rows) {
    if (!acc) { console.log(`  ${label.padEnd(22)} ${(share * 100).toFixed(0).padStart(4)}%   (too few of one class)`); continue; }
    console.log(`  ${label.padEnd(22)} ${(share * 100).toFixed(0).padStart(4)}%  ` +
      NAMES.map(k => (acc[k] * 100).toFixed(0).padStart(4) + '%').join(' '));
  }
}
console.log('');
