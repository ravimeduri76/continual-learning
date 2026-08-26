# Five Dials of Plasticity — the unified framework

**Artifact:** https://claude.ai/code/artifact/ba5d0b43-d59b-49e0-b68e-d9919f032db7
**Demonstrator:** https://claude.ai/code/artifact/8e6b9e15-319d-4c6e-add1-08abb8df60aa
**Written:** 26 Aug 2026

## The master equation

Every continual learner is one update rule with five knobs. For each parameter tier *k*:

```
θₖ ← θₖ − ηₖ · Πₖ [ ∇_θₖ ℒ(θ ; 𝒟ₜ ∪ ℛₜ) + λₖ Mₖ (θₖ − aₖ) ]
```

| Dial | Symbol | What it controls |
|---|---|---|
| **Locus** | θₖ, Πₖ | which parameters may move, along what projection |
| **Pace** | ηₖ | how fast — equivalently N_eff, how many past examples it still answers to |
| **Anchor** | aₖ | what it is pulled back toward (origin / previous solution / pretrained init) |
| **Geometry** | Mₖ | in which directions the pull acts (I / Fisher / hard mask) |
| **Evidence** | 𝒟ₜ ∪ ℛₜ | what the gradient is permitted to see (stream / stream + replay) |

Corner cases: freezing is ηₖ = 0. Prototype/nearest-centroid is the closed-form corner.
**In-context learning** is a tier whose Pace is one forward pass and whose Anchor is
θ_pretrained *exactly* — Dherin et al. (arXiv:2507.16003) prove context acts as a rank-1
update `ΔW = W·A(C,x)·xᵀ/‖x‖²` on the MLP's first weight matrix.

## Two findings worth carrying forward

**1. FTRL-Proximal and EWC are the same update.** Both are `argmin[ ℒ_new + ½‖θ − anchor‖²_M ]`.
FTRL's per-coordinate stiffness `σ ∝ √Σg²`, AdaGrad's preconditioner, the Kalman posterior
precision `P⁻¹ = Σxxᵀ/R`, and EWC's diagonal Fisher `Fᵢ = E[(∂log p/∂θᵢ)²]` are the *same
statistic* — the second moment of the gradient — under four job titles: an adaptive learning
rate, a posterior precision, a synaptic importance. Google shipped elastic weight consolidation
in ad-click prediction (KDD 2013) four years before the PNAS paper. Synaptic Intelligence is the
path-integral (AdaGrad-shaped) sibling of EWC's endpoint Fisher.

**2. Locus beats Geometry.** The field spent 2016–2020 on the Geometry dial (EWC, SI, MAS) and
it produced the least — pure weight-space regularisation sits near chance in class-incremental
Split-MNIST (van de Ven et al., *Nat. Mach. Intell.* 2022). The field is spending 2024–2026 on
Locus and it is producing the biggest numbers: sparse memory finetuning (arXiv:2510.15103) cuts
NaturalQuestions degradation from **−89% (full FT) → −71% (LoRA) → −11%**, purely by changing
which parameters are eligible to move.

## The five invariants

1. **Conservation.** Within one tier, stability and plasticity trade against a fixed evidence
   budget, optimum at `N_eff* ∝ (σ²/drift²)^⅓`. The only escape is to *partition* — many tiers,
   many clocks. Judge a method by whether it improves the compromise (marginal) or changes the
   partition (potentially large).
2. **Anchoring.** Regularise toward your previous solution, not the origin, weighted by the
   evidence that built it, and anisotropically.
3. **Locality.** Forgetting ∝ `⟨∇ℒ_old, ∇ℒ_new⟩`. Shrink the overlap and forgetting collapses
   faster than any penalty can achieve. This is also why pretraining scale reduces forgetting
   for free (Ramasesh et al., ICLR 2022 — representations become more orthogonal).
4. **Exactness excludes adaptivity.** Conjugate Bayes and Kalman with Q=0 retain everything and
   adapt to nothing. If you need both, get them from *different components*.
5. **The consolidation gap — open.** Every fast mechanism is transient (ICL discarded at
   eviction, TTT at end of sequence, agent scratchpad at end of episode); every persistent
   mechanism is destructive (finetuning, model editing, even SEAL's self-edits degrade earlier
   tasks as they accumulate). Nothing moves a trace from fast to slow without collateral damage.
   This is what sleep does and what no deployed system does. **It is the problem worth working on.**

## State of play, August 2026

- **Nothing beats putting it in the context.** CL-Bench (arXiv:2606.05661): naive full-context ICL
  25.4% normalized gain vs purpose-built ACE at 8.6%; memory systems ranked *below* the naive
  baseline while costing more. Best configuration captured ~25% of available headroom.
- **External memory relocates the problem** — bounded context means old and new memories compete
  at retrieval time (arXiv:2604.27003).
- **When, not whether** — weight updates are necessary for discrete fact revision and online RL
  with clear reward; non-parametric methods suffice under noisy temporal drift; prompt-based
  methods introduce "catastrophic memorizing" (arXiv:2607.07847).
- Most promising architectural proposal: **Nested Learning / Continuum Memory System**
  (arXiv:2512.24695, NeurIPS 2025) — a spectrum of modules each updating at its own frequency.
  Research scale, single group, no independent replication.

Full citation list is in the artifact's Sources section.
