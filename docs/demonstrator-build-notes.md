# The Forgetting Machine — build notes and measured results

**Live demonstrator:** https://ravimeduri76.github.io/continual-learning/

## What it is

Three modes — Shapes (vision), Flags (vision + text), People (text). The **default (public) flow is
short**: 6 like/pass answers → two predictions shown up front → done. Five learners see identical
frozen embeddings; four differ by exactly one dial, the fifth runs two clocks.

**The prediction.** After the 6 answers, the mean P(yes) across all five learners is ranked over
every item held out of train/test, and the two extremes are shown plainly — the item they most
expect you to like and the one to pass on, with confidence. No blind-answer-then-grade, no
self-congratulation; the player judges the read, and a 1–5 rating captures it. On the real
embeddings the top/bottom pick is right ~90–98% of the time (`predict_demo.js`: 91/77/68% liked by
mode from as few as five answers).

**The deeper path (opt-in).** A link continues into the original long protocol — 6 held-out
predictions → the question silently changes → 8 more answers → 6 more predictions → a scorecard
ending in live dials that replay the whole answer history. This is where drift, backward transfer,
and the two-clock/Law I demonstration live; it is no longer forced on a casual visitor.

## The four learners (as dial settings)

| Learner | Dial that differs | Defaults |
|---|---|---|
| Online logistic (SGD) | the control — no anchor, no replay | η=0.9, 4 steps |
| FTRL-Proximal | Anchor + Geometry | α=1.2, β=1, λ₁=0.004, λ₂=1, 4 steps |
| Prototype centroid | Pace = one shot, closed form | γ=1.0, τ=6 |
| Replay + Fisher anchor | Evidence + Anchor + Geometry | η=0.7, λ=0.12, 8 steps, γ_F=0.92 |
| *(toggle)* Head + rank-2 adapter | Locus widened — reproduces LP-FT distortion | η_A=0.30, factors clipped to unit ball |

## Embeddings

Frozen CLIP RN50-quickgelu (open_clip, CC12M checkpoint from the mlfoundations GitHub release —
HuggingFace and download.pytorch.org may be blocked in some environments). PCA to 24d (shapes,
people) / 16+16d (flags: image channel ‖ text channel), mildly whitened, L2-normalised, int8
quantised. Bundle is 557 KB, page is 606 KB total.

**Separability check** (within-group vs between-group cosine, bigger gap = easier):

```
shapes/family   0.615 / -0.081  gap 0.697      flags/continent 0.413 / -0.084  gap 0.497
shapes/count    0.270 / -0.977  gap 1.247      flags/bands     0.220 / -0.450  gap 0.671
shapes/fill     0.327 / -0.322  gap 0.650      people/field    0.450 / -0.131  gap 0.582
                                               people/era      0.125 / -0.885  gap 1.010
```

## Which preference concepts are actually learnable from 9 examples

Held-out accuracy, 30 seeds, best learner:

- **Shapes** — stars & crosses 85%, round things 85%, polygons 82%, groups of three 82%,
  monochrome 74%; colour-based concepts 67–71%.
- **Flags** — has green in it 82%, Europe 77%, mostly green 71%, Africa 71%;
  horizontal bands 61% (the weakest — CLIP's flag PCA does not foreground band orientation).
- **People** — sport 77%, science & tech 73%, politics 69%, musicians 69%;
  era and "from the Americas" are near chance at n=9.

## Measured drift behaviour (24 simulated players, real embeddings)

FLAGS, "has green in it" → "European flags":

```
                          SGD   FTRL  PROTO REPLAY
old concept, before       81%    81%    84%    81%
old concept, after        54%    58%    62%    70%
new concept, after        59%    60%    55%    49%
backward transfer        -27%   -23%   -23%   -11%
```

PEOPLE, "science & tech" → "sports figures": replay best retention (−4% BWT), SGD/FTRL best on
the new concept (68–70%). SHAPES is milder but the same ordering.

**This is the headline result: replay retains best and adapts worst; plain SGD does the exact
opposite. Law I (conservation), made clickable.**

## Two-clock learner — the partition of Law I (added after the initial import)

A fifth learner, `TwoClock` in `learners.js`, built to *demonstrate* the escape the framework
only asserted. It partitions the predictor into two tiers of **different geometry**, so it does
not reduce to a single head:

```
score(x) = τ · (cos(x, μ₊) − cos(x, μ₋))   ← SLOW: closed-form centroid, γ=1, never forgets
         + wF·x + bF                          ← FAST: leaky logistic, η=0.9, ρ=0.5 → N_eff = 2
```

**Why this shape and not the obvious one.** The first attempt was a slow body + fast head that
were *both linear in x* and summed: `w_s + w_f`. That collapses to one weight vector — one clock
in disguise — and the averaged sweep confirmed it sits *inside* the frontier, dominated by FTRL.
A centroid score is nonlinear in `x` (normalisation, separate ± means), so slow+fast here is a
genuine partition. This is the lesson of Invariant 1 in code: the escape requires the tiers to be
irreducible, not just differently-tuned.

**Measured** (`twoclock_demo.js`, drift protocol averaged over 200 independent worlds — a single
16-click draw is far too noisy at 2.5% test granularity to read a frontier):

```
learner                 A before   A after    B after   retention × adaptation
Online logistic (SGD)     71.9%     63.5%     67.5%        0.428
FTRL-Proximal             77.1%     67.6%     69.1%        0.467
Prototype centroid        80.7%     71.0%     67.0%        0.476
Replay + Fisher           80.3%     76.9%     61.3%        0.472
Two-clock (slow+fast)     80.7%     70.6%     67.6%        0.477   ← best product, non-dominated
```

**Honest reading.** The two-clock is Pareto **non-dominated**, strictly **dominates the SGD
control**, ties Prototype for **fastest initial learning**, and has the **highest
retention×adaptation product** — the only learner top-group on *both* axes, while every
single-clock learner is strong on one and weak on the other. It does **not** blow the frontier
open: at 16 clicks with linear readouts it *joins* the frontier rather than pushing it outward.
That is faithful to the framework — the Continuum Memory System escape (arXiv:2512.24695) shows
its largest gains at depth/scale, not at this size. It also does not touch Invariant 5: the two
traces never merge, so nothing is claimed about the consolidation gap.

## Verification performed

- `reference.py` — FTRL's materialised `w` equals the numerical argmin of its stated objective
  to 4e-08 (scipy Powell); JS matches numpy to 7e-16 after 25 examples × 4 steps. Prototype
  scores match to 1e-12.
- `e2e.py` — headless Chromium playthrough of all three modes in both themes, zero page errors,
  dial controls exercised.
- `scan.js` / `validate.js` — the tables above.

## Design decisions worth remembering

- **Every learner is a pure function of the answer log.** Changing a dial replays history from
  scratch; nothing is refitted incrementally. This is what makes the end-of-game dial panel
  honest rather than decorative.
- **Test items are never trained on**, so backward transfer measures generalisation on the old
  question rather than memorisation of it.
- **Farthest-point sampling** picks which items to show — spreads questions across the embedding
  space instead of clustering them. Herding, in miniature.
- **Page-Hinkley, not ADWIN**, for the live drift gauge: at n≈17 ADWIN's cut threshold is
  ε≈0.83, larger than any error gap a game this short can produce. The page reports ADWIN's
  threshold anyway, as a teaching moment about what guarantees cost.
- Accuracy at n=6 only moves in 17% steps, so the scorecard also reports **mean P(your answer)**,
  a proper scoring rule that separates learners which tie on accuracy.

## Obvious next steps

1. **Transformer mode.** The natural fourth mode: show the player a prompt, let an actual small
   LM predict, and contrast in-context adaptation (context grows) against a LoRA-style update
   (weights move). That would put all three regimes in one artifact.
2. ~~**A second timescale.**~~ **Done** — see "Two-clock learner" above (`learners.js`,
   `twoclock_demo.js`), and now raced live in the game as a first-class fifth learner (teal).
   It rides on its own defaults — the η/λ/γ dials cannot reach it, which is the point of a
   partition — and the scorecard note now resolves the conservation law onto it.
3. **Uncertainty sampling** as an optional item selector, to show how much of the sample
   efficiency gap is the learner and how much is the curriculum.
