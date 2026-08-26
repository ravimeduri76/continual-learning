# The Forgetting Machine — build notes and measured results

**Artifact:** https://claude.ai/code/artifact/8e6b9e15-319d-4c6e-add1-08abb8df60aa
**Source archive:** `Continual learning src.tar.gz` (alongside this file)

## What it is

Three modes — Shapes (vision), Flags (vision + text), People (text) — each running the same
protocol: 9 like/pass answers → 6 held-out predictions → the question silently changes → 8 more
answers → 6 more predictions → scorecard. Four learners see identical frozen embeddings and
differ by exactly one dial each. The scorecard ends in live dials that replay the player's whole
answer history under different settings.

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
HuggingFace and download.pytorch.org are both blocked from the sandbox). PCA to 24d (shapes,
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
2. **A second timescale.** Right now every learner has one clock. Adding a two-tier learner —
   fast head + slow anchored body, updating at different frequencies — would demonstrate the
   partition escape from Law I directly, rather than only asserting it.
3. **Uncertainty sampling** as an optional item selector, to show how much of the sample
   efficiency gap is the learner and how much is the curriculum.
