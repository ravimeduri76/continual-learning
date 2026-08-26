# Continual Learning — from classic ML to Transformers

A study + demonstrator on *how* systems can keep learning after deployment,
walking the same problem across three regimes:

1. **Classic ML** — parametric models (linear/logistic) don't self-update; the
   data-scientist move is *retrain on new evidence*. Trees add rules fast (lift,
   support). This is the baseline intuition.
2. **Deep nets** — transfer learning, gradual unfreezing, Follow-the-Leader /
   Follow-the-Regularized-Leader.
3. **Transformers** — heavy pre-training precisely *because* they don't learn
   continuously; growing context vs. a LoRA update as the two escape hatches.

The goal is **not to solve** catastrophic forgetting, but to give insight into
how it *could* be approached — and to make it tangible with a game.

## The unified framework — "Five Dials of Plasticity"

One update rule, five knobs, ~40 named methods (1960 → 2026) mapped onto knob
settings:

```
θₖ ← θₖ − ηₖ · Πₖ [ ∇ℒ(θ; 𝒟ₜ ∪ ℛₜ) + λₖ Mₖ (θₖ − aₖ) ]
```

- **Locus** — what may move (layers / activations / weights / a tuning window)
- **Pace** — how fast (η, N_eff)
- **Anchor** — what θ is pulled toward (aₖ)
- **Geometry** — in which directions (Mₖ, Πₖ)
- **Evidence** — what the gradient may see (data ∪ replay ℛₜ)

Key claims to preserve (verified in the source session, re-verify on merge):
- FTRL-Proximal and EWC are the *same* argmin update; stiffness is the second
  moment of the gradient (AdaGrad / Kalman posterior precision / diagonal Fisher).
- LP-FT (Kumar et al., ICLR 2022) gives the freezing instinct a theorem; sparse
  memory finetuning (2026) shows Locus beats Geometry for retention.

## The game — "The Forgetting Machine"

Three modes, ask 5–10 like/unlike questions, then predict the next items:

| Mode | Representation |
|------|----------------|
| Shapes | Computer vision (procedurally generated → known ground truth) |
| Flags | Vision **+** text embeddings (image ‖ country-name in one shared space) |
| Persons/Celebrities | Text embeddings |

**Embeddings: real everywhere** — frozen CLIP RN50 (CC12M) throughout.

**Four learners race side-by-side on the same 8 clicks** (the core of the demo —
watch them disagree):
- Online logistic + FTRL
- Frozen backbone + last layer
- Prototype / nearest-centroid
- Replay buffer + drift penalty

Conservation-law finding to reproduce: no single winner — Replay retains best but
adapts worst; plain SGD does the opposite.

## Deliverable sequence

1. **Framework doc** (Five Dials of Plasticity) — [`docs/five-dials-framework.md`](docs/five-dials-framework.md)
2. **Game** (The Forgetting Machine) — [`src/`](src/), build notes in
   [`docs/demonstrator-build-notes.md`](docs/demonstrator-build-notes.md)

## Running the demonstrator

No CLIP needed — the pre-computed embeddings ship in `src/assets/bundle.json`:

```
python3 src/build.py     # regenerates src/game.html (open it in a browser)
node src/test_learners.js  # sanity-check the five learners + drift detectors
```

Reproducing the embeddings from scratch (needs the 408 MB CLIP checkpoint) is
documented in [`src/README.md`](src/README.md).

## Status

- **Two-clock learner** (partition escape from Law I) — **done**. Slow centroid +
  fast leaky logistic; a genuine irreducible partition (`src/learners.js`),
  measured over 200 worlds in `src/twoclock_demo.js`, and raced live in the game
  as a first-class fifth learner. See [`docs/demonstrator-build-notes.md`](docs/demonstrator-build-notes.md).
- **Next: transformer mode** — a fourth game mode contrasting a growing context
  against a LoRA-style weight update, to put classic ML, deep nets, and
  transformers in one artifact.

## Provenance

Bootstrapped from a Claude Cowork session. Original artifacts were unpacked from
`incoming/` into `docs/` (the two markdown docs) and `src/` (the tarball). The
tarball is retained in `incoming/` as the pristine source.
