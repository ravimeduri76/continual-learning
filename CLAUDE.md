# CLAUDE.md — continual-learning

A framework and a playable demonstrator for continual learning across three regimes:
classic ML → deep networks → transformers. The point is not to solve catastrophic
forgetting but to show that forty years of methods are one update rule with five knobs.
The public flow is short — 6 answers, two predictions — with an opt-in deeper path that makes
the retention/adaptation trade-off (drift → backward transfer) something a person can feel.

Two deliverables:

- Framework — `docs/five-dials-framework.md`
- Game — the playable demonstrator, deployed to https://ravimeduri76.github.io/continual-learning/

## The shared vocabulary

All code and prose in this repo names things after the five dials. Use the same words.

```
θₖ ← θₖ − ηₖ · Πₖ [ ∇ℒ(θ ; 𝒟ₜ ∪ ℛₜ) + λₖ Mₖ (θₖ − aₖ) ]
```

| Dial | Symbol | Meaning |
|---|---|---|
| Locus | θₖ, Πₖ | which parameters may move, along what projection |
| Pace | ηₖ | how fast — equivalently `N_eff`, how many past examples it still answers to |
| Anchor | aₖ | what it is pulled back toward (origin / previous solution / pretrained init) |
| Geometry | Mₖ | in which directions the pull acts (I / diagonal Fisher / hard mask) |
| Evidence | 𝒟ₜ ∪ ℛₜ | what the gradient may see (stream only / stream + replay) |

Freezing is `ηₖ = 0`. Prototype/nearest-centroid is the closed-form corner. In-context
learning is a tier whose Pace is one forward pass and whose Anchor is θ_pretrained exactly
(arXiv:2507.16003 — context acts as a rank-1 update `ΔW = W·A(C,x)·xᵀ/‖x‖²` on the MLP).

Two claims the repo is built to support, both load-bearing:

- **FTRL-Proximal and EWC are the same update.** Both are `argmin[ℒ_new + ½‖θ−anchor‖²_M]`.
  FTRL's stiffness `σ ∝ √Σg²`, AdaGrad's preconditioner, the Kalman posterior precision, and
  EWC's diagonal Fisher are the *same statistic* — the second moment of the gradient.
- **Locus beats Geometry.** Sparse memory finetuning cuts NQ degradation −89% → −11% purely by
  restricting which parameters may move (arXiv:2510.15103), where a decade of penalty design
  achieved much less.

## Layout

```
docs/five-dials-framework.md      the framework, prose
docs/demonstrator-build-notes.md  build decisions + measured results
src/learners.js                   THE ENGINE — five learners, Page-Hinkley, ADWIN threshold. No DOM.
src/game.template.html            the UI. Edit this, never game.html.
src/build.py                      inlines learners.js + assets/bundle.json → game.html
src/gen_{shapes,flags,people}.py  item generation with known generative factors
src/embed.py                      frozen CLIP RN50 → PCA → int8 → assets/bundle.json
src/assets/bundle.json            committed. The game runs from this; no model needed at play time.
src/weights/rn50-cc12m.pt         408 MB, gitignored, only needed to re-run embed.py
incoming/                         original source drop, kept for provenance
```

## Commands

```bash
python3 src/build.py                 # regenerate src/game.html (gitignored, always rebuildable)
node src/test_learners.js [dim]      # synthetic clustered stream, drift, LP-FT probe
node src/scan.js                     # which preference concepts are learnable from 9 examples
node src/validate.js                 # full drift protocol on the real embeddings
node src/predict_demo.js             # the original brief: predict the 6th liked / 7th unliked
node src/twoclock_demo.js            # the partition escape from Law I, 200 worlds
python3 src/reference.py             # FTRL closed form vs numerical argmin; JS vs numpy
python3 src/e2e.py <mode> <theme>    # headless playthrough + screenshots (needs playwright)
```

`embed.py` is the only step that needs torch and the checkpoint. Everything else runs from
the committed bundle.

## Invariants — do not break these

1. **Every learner is a pure function of the answer log.** Changing a dial replays the whole
   history from scratch; nothing is refitted incrementally, nothing is cached. This is what
   makes the end-of-game dial panel honest rather than decorative. If you add a learner, it
   must be reconstructible from `(log, dials)` alone.
2. **Test items are never trained on.** Backward transfer must measure generalisation on the
   old question, not memorisation of it. Accuracy is always recomputed from a replayed model,
   never from a cached prediction.
3. **`src/game.html` is generated.** Edit `game.template.html` and rebuild. It is gitignored.
4. **All learners see identical embeddings.** Any difference in outcome must be attributable to
   a dial, or the demonstration is worthless.
5. **Report what a method actually bought.** The repo's credibility rests on the negative
   results as much as the positive ones — see the two-clock reading below.

## Measured baselines (regression targets)

`node src/predict_demo.js` — 5 answers, predict the next liked and next unliked item,
400 draws × 4 concepts per mode:

```
             6th liked   7th unliked   both      random "both"
shapes          91%          98%        89%          21%
flags           78%          92%        72%          23%
people          57%          74%        36%          24%   (60 business leaders; base admire-rate ~40%)
```

`node src/twoclock_demo.js` — 200 worlds, 9 clicks on concept A → drift → 7 on concept B:

```
                       A before  A after  B after   retention × adaptation
Online logistic (SGD)    71.9%    63.5%    67.5%        0.428
FTRL-Proximal            77.1%    67.6%    69.1%        0.467
Prototype centroid       80.7%    71.0%    67.0%        0.476
Replay + Fisher          80.3%    76.9%    61.3%        0.472
Two-clock (slow+fast)    80.7%    70.6%    67.6%        0.477   ← Pareto non-dominated
```

Read this honestly: replay retains best and adapts worst, SGD does the opposite — that split
is Law I (conservation). Two clocks reach the frontier and strictly dominate the control, but
they **do not blow the frontier open**. At 16 clicks with linear readouts they cannot; that
needs depth and scale. Do not oversell it.

Concept learnability from 9 examples (`scan.js`), for picking demo concepts: shapes —
stars & crosses 85%, round things 85%, polygons 82%; flags — has green in it 82%, Europe 77%;
people (60 business leaders) — tech-vs-not (89%) and region-Americas (92%) are the strong
axes; sparse ones (Asia, finance) lift less. Roster pruned to recognisable names; no
"polarising" label is shown (it would prime the rating). Near chance: flags/horizontal-bands.

## Gotchas

- **HuggingFace and download.pytorch.org may be blocked in some environments**; the open_clip
  GitHub release is reachable. That is why the checkpoint is `rn50-quickgelu-cc12m` rather than
  a nicer ViT.
- **`cairosvg` does not parse `hsl()`** — `gen_shapes.py` resolves colours to hex in Python.
  Silently renders black otherwise.
- **`game.template.html` carries no doctype/head/body wrapper.** The Pages deploy (and
  `e2e.py`) wrap it into a complete document. Keep it self-contained: inline all CSS/JS and
  embed images as data URIs; only Google Fonts loads externally. Fonts fail silently in
  `file://` previews — expected.
- **int8 decode in the browser**: `atob` gives unsigned bytes; sign-extend with
  `(c << 24) >> 24` before scaling. Getting this wrong yields a page that looks fine and
  learns nothing.
- **Page-Hinkley, not ADWIN**, drives the live drift gauge. At n≈17 ADWIN's cut threshold is
  ε≈0.83, larger than any error gap this protocol can produce. The page reports ADWIN's
  threshold anyway, as a lesson in what guarantees cost. Keep that framing.
- **Six test items means accuracy moves in 17% steps.** Always report mean P(correct answer)
  alongside it; it is continuous and separates learners that tie.

## Open next step

A transformer mode: show a prompt, let a small LM predict, and contrast in-context adaptation
(context grows, weights frozen, discarded at eviction) against a LoRA-style update (weights
move, persists, degrades earlier tasks). That would put all three regimes in one artifact and
close the argument the framework doc opens.
