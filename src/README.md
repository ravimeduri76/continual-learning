# Continual Learning — framework + demonstrator

Two deliverables:

- **Five Dials of Plasticity** — the unified framework (classic ML → deep nets → transformers)
  `../docs/five-dials-framework.md`
- **The Forgetting Machine** — the playable demonstrator
  https://ravimeduri76.github.io/continual-learning/

## The claim

Every continual learner is one update rule with five knobs:

    θₖ ← θₖ − ηₖ · Πₖ [ ∇ℒ(θ; 𝒟ₜ ∪ ℛₜ) + λₖ Mₖ (θₖ − aₖ) ]

  Locus     θₖ, Πₖ        what may move, along what projection
  Pace      ηₖ            how fast — equivalently N_eff, how many past examples it answers to
  Anchor    aₖ            what it is pulled back toward
  Geometry  Mₖ            in which directions the pull acts
  Evidence  𝒟ₜ ∪ ℛₜ       what the gradient is permitted to see

Freezing is ηₖ=0. Prototype learning is the closed-form corner. In-context learning is a tier
whose Pace is one forward pass and whose Anchor is θ_pretrained exactly (Dherin et al. 2025:
context acts as a rank-1 update ΔW = W·A(C,x)·xᵀ/‖x‖² on the MLP).

## Pipeline

    gen_shapes.py     240 SVG specimens with known generative factors
    gen_flags.py      194 national flags (lipis/flag-icons) + derived colour/band metadata
    gen_people.py     187 public figures as CLIP text prompts
    embed.py          frozen CLIP RN50 (open_clip, CC12M) -> PCA -> int8 -> assets/bundle.json
    learners.js       the five learners + Page-Hinkley + ADWIN threshold. No DOM.
    build.py          inlines learners.js + bundle.json into game.html
    e2e.py            headless playthrough, screenshots, console-error check

## Verification

    python3 reference.py     FTRL closed form == numerical argmin (4e-08); JS == numpy (7e-16)
    node test_learners.js    synthetic clustered stream, drift, LP-FT probe
    node scan.js             which preference concepts are learnable from 9 examples
    node validate.js         full protocol on the real embeddings, 24 simulated players
    node twoclock_demo.js    two-clock partition vs the single-clock frontier, 200 worlds
    node predict_demo.js     from 5 answers, predict a liked 6th & unliked 7th item

## Reproducing the embeddings

CLIP weights are NOT in this archive (408 MB). Fetch with:

    mkdir -p weights && curl -L -o weights/rn50-cc12m.pt \
      https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt
    pip install torch open_clip_torch cairosvg pycountry pycountry-convert
    python3 gen_shapes.py && python3 gen_flags.py && python3 gen_people.py
    python3 embed.py && python3 build.py

## Measured behaviour (24 simulated players, real embeddings)

FLAGS, "has green in it" -> "European flags", 9 train / 8 after drift:

                              SGD   FTRL  PROTO REPLAY
    old concept, before       81%    81%    84%    81%
    old concept, after        54%    58%    62%    70%
    new concept, after        59%    60%    55%    49%
    backward transfer        -27%   -23%   -23%   -11%

Replay retains best and adapts worst; plain SGD does the opposite. That split is Law I —
within one tier, retention and adaptation trade against a fixed evidence budget.

## Predict from 5 answers (node predict_demo.js)

Top-1 / bottom-1 of the ranked pool = predicted 6th (liked) & 7th (unliked) item,
400 draws x 4 concepts per mode, best-of-pack learner:

                    6th liked   7th unliked   both   random-pick both
    shapes             91%          98%        89%        21%
    flags              77%          92%        71%        23%
    people             68%          86%        59%        21%

The liked prediction is the hard one (base like-rate ~30%); all three modes clear it
decisively. People is weakest — text descriptors are the least separable, five examples
are genuinely thin there. Holds only when the 5 answers contain both a like and a
dislike, which farthest-point sampling delivers 86-97% of the time.
