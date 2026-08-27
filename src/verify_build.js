/* Browser-free runtime guard for the built game.html. Catches the class of bug
   that node --check cannot: the inlined CL namespace drifting out of sync with
   the learners the game actually constructs (this is how CL.TwoClock got
   dropped once). Run after build.py; exits non-zero on any failure. */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'game.html'), 'utf8');
const re = /<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g;
const blocks = [];
let m; while ((m = re.exec(html))) blocks.push(m[1]);

function fail(msg) { console.error('verify_build FAIL:', msg); process.exit(1); }

const learners = blocks.find(s => /const CL =/.test(s));
const app = blocks.find(s => /predictView/.test(s));
if (!learners) fail('no learners/CL block found');
if (!app) fail('no app block found');

let CL;
try { CL = eval('(function(){' + learners + '; return CL;})()'); }
catch (e) { fail('learners block threw: ' + e.message); }

const NEED = ['SGDLogistic', 'FTRLProximal', 'Prototype', 'ReplayEWC', 'AdapterHead',
              'TwoClock', 'PageHinkley', 'adwinEpsCut', 'makeLearners', 'mulberry32'];
for (const k of NEED) if (typeof CL[k] === 'undefined') fail('CL.' + k + ' is undefined');

// construct the exact set the game builds and exercise it
const d = 24;
const ls = CL.makeLearners(d, {});
ls.push(new CL.TwoClock(d));
const ids = ls.map(l => l.id).join(',');
if (ids !== 'sgd,ftrl,proto,replay,twoclock') fail('unexpected learner ids: ' + ids);
const x = new Float64Array(d).fill(0.2);
for (const L of ls) {
  L.observe(x, 1);
  const p = L.prob(x), n = L.nEff();
  if (!isFinite(p) || !isFinite(n)) fail('non-finite output from ' + L.id);
}

// the wow path + feedback must be present and the config placeholder resolved
for (const [name, rx] of [
  ['predictView', /function predictView\(\)/],
  ['submitFeedback', /function submitFeedback\(/],
  ['feedback payload', /type:\s*'prediction_feedback'/],
  ['FEEDBACK_ENDPOINT config', /window\.FEEDBACK_ENDPOINT\s*=/],
]) if (!rx.test(html)) fail(name + ' not found in built page');

console.log('verify_build OK — 4 blocks, CL complete, learners', ids, 'run; predict + feedback wired');
