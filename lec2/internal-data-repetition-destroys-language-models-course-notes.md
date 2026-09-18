# Internal Data Repetition Destroys Language Models

**Course notes**

> Chudnovsky, Kazdan, Levi, Schaeffer, Denisov-Blanch, He, Donmez, Koyejo, Donoho. *Internal Data Repetition Destroys Language Models.* arXiv:2606.24998v1 \[cs.LG\], 23 June 2026.

**The one-sentence version.** If 10% of your pretraining tokens are duplicates, the damage depends not on *how many* duplicated tokens there are but on *how concentrated* they are — and at the worst concentration you throw away a third of your compute budget for a loss regression you'd barely notice on a plot.

**Prerequisites.** Transformer pretraining basics (Lecture 1), cross-entropy loss in nats, and comfort with power laws. Section 7 additionally assumes ordinary least squares and the trace identities for quadratic forms; it can be skipped on a first pass without losing the empirical thread.

------------------------------------------------------------------------

## 1. The problem

### 1.1 Why repetition is now unavoidable

Pretraining has run out of easy data. The stock of high-quality public text that frontier labs actually want has been largely consumed, which pushes training into multi-epoch territory whether anyone wanted that or not.

The field's response was deduplication. FineWeb-Edu, DataComp-LM, Dolma, and RedPajama-v2 all ship with aggressive dedup and filtering pipelines. The problem is that **aggressive deduplication is not perfect deduplication**. What survives a dedup pass includes near-duplicate documents, boilerplate templates rendered with different content, and semantically redundant pages that no hash-based method will catch. Worse, what counts as a "duplicate" is itself scale-dependent — a larger model recognizes more document pairs as functionally the same.

So the practical question isn't "how do I eliminate repetition" (you can't) but "how much is repetition costing me, and which repetition patterns are worst?"

### 1.2 What was known before

Two prior lines of work bracket the question, and this paper sits between them.

|  | Hernandez et al. (2022) | Muennighoff et al. (2023) | This paper |
|----|----|----|----|
| Repetition regime | Small fraction repeated | **Entire** corpus repeated uniformly | Small fraction repeated |
| Budget discipline | Fixed 300B tokens for *every* model size | Chinchilla-aware | Chinchilla-aware, iso-FLOP sweeps |
| What's varied | Repeat count | Number of epochs | Repeat count at **fixed repeated-token fraction** |
| Damage measured as | Reduction in effective parameter count | Additive overfitting term in the loss | **Compute-Equivalent Gain / Loss** |
| Headline | Non-monotonic damage; induction heads degrade | ~4 epochs are nearly free | Damage peaks at intermediate concentration; up to 33% of compute wasted |

Hernandez et al. is the direct ancestor, and it's worth understanding precisely why its measurement was unsatisfying. Every model in that study trained on 300B tokens regardless of its parameter count. Under Chinchilla-optimal allocation you want roughly 20 tokens per parameter, so a fixed 300B-token budget leaves the large models badly undertrained and the small models grossly overtrained. The models being compared were therefore sitting at wildly different places on the compute–loss curve, which makes "effective parameter count" hard to read across the grid.

There's a second, subtler problem with the effective-parameter framing. Practitioners don't allocate parameters; they allocate FLOPs. Telling an engineer "your duplicates cost you the equivalent of 15% of your parameters" is not an actionable number. Telling them "your duplicates cost you 33% of your compute budget" is.

### 1.3 The gap this paper fills

Three things were missing:

1.  A measurement of repetition damage in **compute units**, under budgets a modern lab would actually use.
2.  A study that holds the *amount* of repeated data fixed and varies only its *structure* — so the effect of concentration is cleanly separated from the effect of volume.
3.  An account of **why** the damage is non-monotonic that doesn't depend on transformer-specific machinery.

**Check yourself.** Why is "the whole corpus repeated for 4 epochs" a fundamentally different question from "10% of tokens are duplicates"? What changes about the model's estimate of the data distribution in each case?

------------------------------------------------------------------------

## 2. The measurement apparatus

This is the part of the paper to understand deeply. The experiments are a sweep; the contribution is the *coordinate system* the sweep is plotted in.

### 2.1 The Chinchilla budget identity

Let $N$ be the parameter count, $T$ the total training tokens, $C$ the training compute in FLOPs. The standard dense-transformer estimate is $6N$ FLOPs per token (roughly: two FLOPs per parameter for the forward multiply–accumulate, times three to cover the backward pass). So:

$$C = 6NT$$

Now introduce an **overtraining multiplier** $\mathrm{OT}$, defined so that $\mathrm{OT} = 1$ means the Chinchilla-optimal 20 tokens per parameter:

$$T = 20 \cdot \mathrm{OT} \cdot N$$

Substituting:

$$C = 6N \cdot 20 \cdot \mathrm{OT} \cdot N = 120 \cdot \mathrm{OT} \cdot N^2 \qquad \text{(1)}$$

Read equation (1) as a *constraint*. Once you fix $(N, \mathrm{OT})$, both your token budget and your FLOP budget are pinned. Anything else you change — including how you arrange duplicates — happens inside a sealed box. That is what makes the comparison honest: every point in a sweep costs exactly the same.

### 2.2 Repetition structure: the knob that actually matters

Fix $f = 0.1$: one tenth of training tokens come from a repeated pool, nine tenths from documents seen once. This fraction is held constant in every experiment. It's chosen to be large enough to produce measurable damage while leaving the bulk of training on unique data, and it matches Hernandez et al. so the two studies can be compared.

Let $D_r$ be the number of **unique** tokens in the repeated pool and $R$ the number of times each repeated document is replayed. The repeated-token budget must add up:

$$fT \approx R D_r \quad\Longrightarrow\quad D_r \approx \frac{fT}{R} = \frac{2 \cdot \mathrm{OT} \cdot N}{R} \qquad \text{(2)}$$

(The last step uses $f = 0.1$ and $T = 20\,\mathrm{OT}N$, so $fT = 2\,\mathrm{OT}N$.)

Equation (2) is the whole conceptual move of the paper, so slow down here. **Turning up $R$ does not make the model train on more repeated material.** The repeated tokens always occupy exactly 10% of the run. Turning up $R$ takes that fixed 10% and squeezes it onto a *smaller* pool of unique documents.

Two views of the same object:

- The **$R$ view** asks: how many times does the model see each repeated document?
- The **$D_r$ view** asks: how big is the repeated corpus?

They're locked together by (2), and the paper calls the pair a **repetition structure**. Same budget, radically different structures:

| Structure         | $R$ | Share of corpus repeated | Repeated tokens |
|-------------------|-------|--------------------------|-----------------|
| Broad and shallow | 10    | 1%                       | 10% of run      |
| Narrow and deep   | 1000  | 0.01%                    | 10% of run      |

Both spend a tenth of the FLOPs on duplicates. The paper's central empirical claim is that these two are **not equally harmful**, and that neither is the worst case.

### 2.3 Compute-Equivalent Gain and Loss

Here's the accounting trick that converts a loss number into a compute number.

First, fit a no-repetition reference curve. Train a clean baseline at each model size with $\mathrm{OT}=1, R=1$ and fit the standard three-parameter Chinchilla form to the six points:

$$L(C) = E + K C^{-\gamma}$$

Term by term:

- $E$ is the **irreducible loss floor** — where the curve asymptotes as $C \to \infty$. It is the entropy the model can never model away: genuine unpredictability in the text plus whatever this architecture and tokenizer structurally cannot capture.
- $K$ sets the vertical scale of the reducible part.
- $\gamma > 0$ is the **rate** at which loss falls with compute. It's the slope you'd see on a log-log plot of $(L - E)$ against $C$.

The fitted values reported are

$$L(C) = 2.365 + 6.647\times10^{5} C^{-0.317} \qquad \text{(6)}$$

Now invert. Given any measured loss $L$, ask how much compute a *clean* run would have needed to get there:

$$C^{\star}(L) = \left(\frac{K}{L - E}\right)^{1/\gamma}$$

And define:

$$\mathrm{CEG} = \frac{C^{\star}(L)}{C_{\text{actual}}}, \qquad \mathrm{CEL} = 1 - \mathrm{CEG} \qquad \text{(7)}$$

Reading the metric:

- $\mathrm{CEG} = 1$ — the repeated run matched the clean reference. No harm done.
- $\mathrm{CEG} = 0.67$ — your run reached a loss a clean run would have hit with **67% of the FLOPs**. You bought a $C_{\text{actual}}$-sized run and received a $0.67\,C_{\text{actual}}$-sized result. $\mathrm{CEL} = 0.33$: a third of the budget evaporated.
- $\mathrm{CEG} > 1$ can happen, and the paper is candid about it: CEG is defined against *this fitted curve*, not against an oracle. A run can beat the fitted reference — for instance because Chinchilla's 20-tokens-per-parameter rule was estimated on a different model family, optimizer, and tokenizer, so it may not be optimal here. Treat $\mathrm{CEG}>1$ as a statement about the reference curve's calibration, not as free lunch.

### 2.4 The formula worth memorizing

If both the repeated run and its clean counterpart sit on the fitted law, (7) collapses to something much more useful than the raw definition:

$$\boxed{\mathrm{CEG} = \left(\frac{L_{\text{base}} - E}{L_{\text{rep}} - E}\right)^{1/\gamma}}$$

Two things fall out immediately.

**First: the exponent $1/\gamma \approx 3.15$ is an amplifier.** A 1% increase in the *reducible* loss $(L-E)$ costs you roughly 3% of your compute. The paper states exactly this sensitivity.

**Second — and this is the part that isn't obvious — the amplification gets worse as models get better.** What you observe on a loss plot is a percentage of *total* loss $L$, but the formula eats percentages of *reducible* loss $L - E$. The conversion factor between them is $L/(L-E)$, and that ratio grows as a model approaches the floor. Chain the two together:

```math
\text{compute cost} \approx \underbrace{\frac{L}{L-E}}_{\text{total} \to \text{reducible}} \times \underbrace{\frac{1}{\gamma}}_{\text{reducible} \to \text{compute}} \times \text{(\% loss regression)}
```

At 34M parameters the fitted curve gives $L \approx 4.81$ and $L - E \approx 2.45$, so the amplifier is about $2.0 \times 3.15 \approx 6$. At 344M, $L \approx 2.93$ and $L - E \approx 0.56$, so the amplifier is about $5.2 \times 3.15 \approx 16$.

**The same visible loss regression is nearly three times more expensive at 344M than at 34M.** This single fact explains the paper's headline — and it explains why the loss-space view of repetition damage systematically understates the bill.

### 2.5 Worked example: the headline number, end to end

Take the largest configuration: $N = 344$M, $\mathrm{OT} = 1$.

**Step 1 — budgets.** $T = 20 \times 3.44\times10^8 = 6.88\times10^{9}$ tokens. $C = 120 \times (3.44\times10^8)^2 = 1.42\times10^{19}$ FLOPs.

**Step 2 — the repeated pool.** $fT = 0.1 \times 6.88\times10^9 = 6.88\times10^{8}$ repeated tokens. At the empirically worst setting $R \approx 155$:

$$D_r = \frac{6.88\times10^8}{155} \approx 4.4\times10^{6}\,\text{unique tokens}$$

Documents are truncated at 2048 tokens, so that pool is **at least ~2,200 documents** and realistically a few thousand. Sit with that: a few thousand documents, each replayed about 155 times, inside a 6.9-billion-token run.

**Step 3 — the clean baseline.** Plug $C = 1.42\times10^{19}$ into (6):

$$L_{\text{base}} = 2.365 + 6.647\times10^5 \times (1.42\times10^{19})^{-0.317} \approx 2.365 + 0.564 = 2.93$$

**Step 4 — the damaged run.** The measured peak sits at $L_{\text{rep}} - E \approx 0.65$ nats, i.e. $L_{\text{rep}} \approx 3.01$. The gap is **0.08 nats — a 2.6% loss increase**, which on a plot spanning 3 to 5 is nearly invisible.

**Step 5 — convert.**

$$\mathrm{CEG} = \left(\frac{0.564}{0.640}\right)^{1/0.317} = (0.881)^{3.155} \approx 0.67$$

A 2.6% loss regression is a **33% compute loss**. That is the paper.

**Check yourself.** Recompute Step 5 for the 34M model, where the worst setting produces roughly a 3.5% loss increase. You should land near $\mathrm{CEL} = 0.19$. Why does the *larger* relative loss regression produce the *smaller* compute penalty?

------------------------------------------------------------------------

## 3. Experimental design

**Models.** Qwen3-style decoder-only transformers trained from scratch at $N \in \{34, 48, 63, 93, 153, 344\}$M. RoPE, RMSNorm, SwiGLU feed-forwards, grouped-query attention, untied input/output embeddings, BF16, FlashAttention-2. Sequence length 2048, vocabulary 151,670, head dimension 128, 32 attention heads and 32 KV heads throughout.

| $N$ | Layers | $d_{\text{model}}$ | $d_{\text{ff}}$ | Non-embedding params | Total params | Non-emb. share |
|----|----|----|----|----|----|----|
| 34M | 3 | 96 | 256 | 4.94M | 34.06M | 14.5% |
| 48M | 4 | 128 | 512 | 9.18M | 48.00M | 19.1% |
| 63M | 5 | 160 | 512 | 14.34M | 62.87M | 22.8% |
| 93M | 6 | 224 | 768 | 25.12M | 93.07M | 27.0% |
| 153M | 9 | 320 | 1024 | 56.04M | 153.11M | 36.6% |
| 344M | 14 | 576 | 1536 | 169.30M | 344.02M | 49.2% |

Note the last column now, because Section 9 comes back to it.

**Corpus and split.** FineWeb-Edu-Dedup, split once with a fixed seed. Roughly 150M tokens held out for evaluation, and critically **the eval split is carved out before any repeated pool is constructed**. Evaluation documents never appear in training, repeated or otherwise. This matters: it means any damage observed cannot be a contamination artifact. It has to come from a distorted effective training distribution.

**Pool construction.** Documents are tokenized, truncated to 2048, and given EOS tokens. A seeded shuffle orders the training split; the repeated pool is the shortest prefix reaching $D_r^\star = fT/R$ tokens, and the non-repeated pool is taken from the documents immediately following. The two are disjoint by construction. Each repeated document is inserted exactly $R$ times, the whole index list is shuffled, and training proceeds. Because cutoffs land on document boundaries, realized pool sizes overshoot targets by at most one document ($\leq 2049$ tokens).

**Optimization.** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.01, gradient clipping 1.0), cosine schedule with warmup ratio 0.2. Peak learning rate is derived from a base of $10^{-6}$ and the token count rather than tuned per model size.

**The grid.** $\mathrm{OT} \in \{0.25, 0.5, 1, 2, 4\}$ and $R$ on an approximately logarithmic grid from 1 up to 20,000 (plots show up to ~3000). 25 completed $(N, \mathrm{OT})$ cells: all five multipliers for 34M through 93M, four through $\mathrm{OT}=2$ for 153M, and $\mathrm{OT}=1$ only for 344M. **One training run per cell** — no seed replicates. Reported losses are final-checkpoint, not best-checkpoint.

------------------------------------------------------------------------

## 4. Result 1 — eval loss is non-monotonic in the repeat count

Hold $f$ and $C$ fixed, sweep $R$ along the iso-FLOP curve, and plot eval loss. The curve **rises, peaks at some intermediate $R$, then partially comes back down**.

The raw numbers: across the completed sweeps the maximum sits 1.0% to 4.2% above the corresponding no-repetition baseline (median 3.1%). Measured as prominence over the higher of the two endpoints — no repeats, and the largest $R$ tested — the peaks are 0.7% to 2.7% (median 1.8%). The peaks are often broad, which is why the authors fit smooth curves rather than trusting the discrete argmax on a coarse grid.

### Why a peak? Three regimes

This is the intuition to actually retain.

**Small $R$ — broad and shallow.** Each repeated document appears two or three times among billions of tokens of fresh text. From the model's perspective this is barely distinguishable from ordinary data. Almost no damage.

**Huge $R$ — narrow and deep.** A handful of documents, each replayed thousands of times. The model memorizes them cold. Crucially, the pool is small enough that this memorization can be *quarantined* — a modest slice of capacity stores those documents nearly verbatim, and the rest of the network goes on modeling the general distribution. The paper's term for this in the theory section is a **memorize-and-isolate fixed point**. Damage recovers, though the recovery is flatter in some sweeps.

**The bad middle.** A pool large enough that it *cannot* be cheaply quarantined, replayed often enough that it *dominates* the gradient signal. Too big to wall off, too loud to ignore. The model's estimate of the general data distribution gets bent toward a sample that isn't representative — and the eval set, remember, was never in there.

**Analogy that stays honest.** You have a fixed number of study hours, a tenth of which goes to flashcards. Three cards seen 5,000 times each: you learn them perfectly, they take up almost no room, and the rest of your studying is untouched. Three thousand cards seen twice each: that's just more reading. Three hundred cards seen 155 times each: now you've spent a real fraction of your effort drilling a set that's big enough to crowd out your general understanding but too big to have simply memorized and set aside. You come out of the exam worse than either extreme.

Use this analogy to catch yourself. If you ever find yourself saying "more repeats is always worse," the 5,000-repeats case should stop you.

**Check yourself.** Two labs both report that 10% of their pretraining tokens were duplicates. Lab A's duplicates are 1% of the corpus repeated 10×; Lab B's are 0.01% repeated 1000×. Based on this section alone, can you say which wasted more compute? What additional number would you need?

------------------------------------------------------------------------

## 5. Result 2 — where the peak sits, as a function of scale

Knowing that a peak exists is only useful if you can predict where. The procedure: for each $(N, \mathrm{OT})$ sweep, fit a Gaussian in $\log_{10} R$ to the eval-loss curve, and take its centre as $R^{\text{peak}}$. Then regress $R^{\text{peak}}$ on $N$ and on $C$ in log-log space.

$$R^{\text{peak}} = 2.31\times10^{10} N^{-0.96}, \qquad D_r^{\text{peak}} = 7.58\times10^{-10} N^{1.84} \qquad \text{(4)}$$

$$R^{\text{peak}} = 1.47\times10^{7} C^{-0.25}, \qquad D_r^{\text{peak}} = 5.49\times10^{-12} C^{0.93} \qquad \text{(5)}$$

**The trend in words: bigger models are hurt most by *fewer* repeats of a *larger* pool.**

Two concrete anchors from the sweep:

| Model | $R^{\text{peak}}$ | $D_r^{\text{peak}}$      |
|-------|---------------------|----------------------------|
| 34M   | ≈ 1400              | ≈ $5\times10^4$ tokens   |
| 344M  | ≈ 155               | ≈ $4.5\times10^6$ tokens |

A 10× increase in model size moves the worst-case pool up by roughly 90×.

The interpretation the authors offer is that this tracks **memorization capacity**. A bigger model can absorb a bigger pool before that pool stops being harmlessly quarantined. Note that $D_r^{\text{peak}} \propto N^{1.84}$ grows *faster* than the compute budget's linear-in-$N$ repeated-token allowance $fT = 2\,\mathrm{OT}N$ — memorization capacity outruns compute.

**Training duration barely matters.** The five $\mathrm{OT}$ sweeps largely fall on the same trend in $N$. Overtraining shifts the *level* of the damage but not the *location* of the worst structure.

### The caveat the authors flag themselves

Because $D_r^{\text{peak}}$ grows like $N^{1.84}$ while the repeated-token budget grows like $N$, the two must eventually cross. Past that point the "predicted" peak pool is bigger than the entire repeated budget, which implies $R < 1$ — nonsense. Setting $7.58\times10^{-10}N^{1.84} = 2N$ puts the crossover somewhere around $N \sim 10^{11}$.

So: **do not use (4) as a predictive tool at frontier scale.** Read it as a statement about a trend — that memorization capacity grows faster than compute within the tested range — not as a formula to extrapolate. The authors say as much.

**Check yourself (worth doing).** Equations (2) and (4) are not independent: $D_r = 2\,\mathrm{OT}N / R$. If $R^{\text{peak}} \propto N^{-0.96}$, what exponent does that force on $D_r^{\text{peak}}$? Compare to the fitted 1.84. What does the mismatch tell you about how these two power laws were fit, and would you expect it to matter more in-range or out-of-range?

------------------------------------------------------------------------

## 6. Result 3 — a 2% loss bump is an O(1) compute loss

Feeding the measured losses through (7) gives the paper's punchline. At $\mathrm{OT}=1$, the worst repeat setting produces:

| $N$   | 34M  | 48M  | 63M  | 93M  | 153M | 344M     |
|---------|------|------|------|------|------|----------|
| **CEL** | 0.19 | 0.19 | 0.21 | 0.21 | 0.26 | **0.33** |
| **CEG** | 0.81 | 0.81 | 0.79 | 0.79 | 0.74 | **0.67** |

Two patterns, and Section 2.4 already explains both.

**The damage grows with model size.** Not because larger models suffer larger loss regressions — they don't, the 344M regression is *smaller* in percentage terms than the 34M one — but because larger models sit closer to the irreducible floor, where the amplifier $\frac{L}{L-E}\cdot\frac{1}{\gamma}$ is much bigger.

**Overtraining changes the level, not the location.** Varying $\mathrm{OT}$ moves the CEG curves up and down but leaves the worst-case $R$ roughly where it was. The dangerous repetition structure is a property of the model, not of how long you train it.

### How much should you trust 0.33?

The authors work through this carefully, and you should be able to reproduce the reasoning.

The factor $(L-E)^{-1/\gamma}$ **diverges as $L \to E$**. With $\gamma \approx 0.32$, a 1% error in $(L-E)$ becomes a ~3% error in CEG. So the reliability of any CEG number depends on how far the run sits above the fitted floor.

- The 344M peak run sits at $L - E \approx 0.65$ nats — comfortably clear of the floor. The headline 33% survives leave-one-out perturbations of the scaling-law fit.
- The **smaller** models live closer to $E$ and inherit correspondingly larger uncertainty. Their CEL values are the shakier ones.

On top of that, the three-parameter Chinchilla form is fit to six points, which is right at the edge of identifiability for that functional form. The authors' own stance is the right one to adopt: **treat the absolute CEG/CEL values as point estimates and the qualitative non-monotonicity as the robust finding.**

------------------------------------------------------------------------

## 7. Why — a statistical model with no transformer in it

A reasonable objection to everything above: maybe the peak is a quirk of attention, or of depth, or of Adam's interaction with repeated gradients. The authors' answer is to reproduce the same qualitative phenomenon in a model with none of those ingredients.

### 7.1 Setup

Inputs $x \sim \mathcal{N}(0, I_p)$ in $\mathbb{R}^p$. Labels are **noiseless**: $y = x^\top \beta$ for a fixed $\beta$.

The twist is that the learner only gets to see the first $m < p$ coordinates. Split $x = (x_{\text{in}}, x_{\text{out}})$ and $\beta = (\beta_{\text{in}}, \beta_{\text{out}})$. The model is **misspecified**: real predictive signal lives in coordinates it can't observe.

The training set has $n$ unique rows plus $d$ rows each duplicated $r$ times, for $n + rd$ total.

Fit OLS on the observed coordinates. The estimator decomposes as

$$\hat{\beta}_{\text{in}} = \beta_{\text{in}} + a_r, \qquad a_r = (X_{\text{in}}^\top X_{\text{in}})^{-1} X_{\text{in}}^\top X_{\text{out}}\beta_{\text{out}}$$

$a_r$ is the **aliasing term**: the damage done when the fit tries to explain unobserved signal using observed coordinates. Finite-sample correlations between $x_{\text{in}}$ and $x_{\text{out}}$ get mistaken for real structure. With no duplication, those correlations are independent across rows and average away. Duplication breaks that.

### 7.2 The crux: $r$ versus $r^2$

Here is the single most important step in the theory, and it's genuinely simple.

**The $r$ copies of a duplicated document share one realization of the unobserved features.** They are not $r$ independent draws that happen to look alike — they are literally the same row, including the same hidden part.

Formally, conditioning on $X_{\text{in}}$, the covariance of $z = X_{\text{out}}\beta_{\text{out}}$ is block-diagonal:

$$\Sigma_r = I_n \oplus \bigoplus_{i=1}^{d} \mathbf{1}_r \mathbf{1}_r^\top \qquad \text{(8)}$$

Unpack it. The unique rows contribute an $n \times n$ identity: independent hidden features, as you'd hope. Each duplicated document contributes an $r \times r$ **all-ones matrix** — every entry 1, because those $r$ rows share a hidden realization exactly. Distinct duplicated documents remain uncorrelated with each other.

*Trace it by hand.* With $n=2$, $d=1$, $r=3$, $\Sigma_r$ is the $5\times5$ matrix that is $I_2$ in the top-left and a $3\times3$ block of all ones in the bottom-right. Its trace is $2+3=5=n+rd$. The all-ones block is rank one with a single eigenvalue equal to $r$.

Now define the Gram matrices $C_u = X_{u,\text{in}}^\top X_{u,\text{in}}$ and $C_d = X_{d,\text{in}}^\top X_{d,\text{in}}$ over *unique* rows. Then:

$$X_{\text{in}}^\top X_{\text{in}} = C_u + rC_d \qquad\text{but}\qquad X_{\text{in}}^\top \Sigma_r X_{\text{in}} = C_u + r^2 C_d$$

**That extra factor of $r$ is the entire story.** The duplicated rows enter the normal equations linearly in $r$, but they enter the *error* quadratically in $r$, because their errors are perfectly correlated instead of averaging out.

*Verify it yourself with $d=1$.* Let the duplicated document have observed row $v$. Then $C_d = v^\top v$. The duplicated block of $X_{\text{in}}^\top X_{\text{in}}$ is $r$ copies summed: $r v^\top v$. But sandwiching the all-ones block sums over all $r \times r$ pairs: $r^2 v^\top v$. Done.

**The analogy.** Ten independent witnesses give you ten noisy accounts; averaging them cancels error. One witness whose statement you photocopy ten times gives you the *same* error ten times — and a least-squares fit reads it as ten-fold corroboration. Duplication doesn't add information; it adds *confidence in one particular mistake*.

### 7.3 Closed-form risks

Conditioning on $X_{\text{in}}$ and taking expectations over the hidden coordinates:

$$\mathbb{E}\left[L_{\text{train}} \mid X_{\text{in}}\right] = \frac{\|\beta_{\text{out}}\|_2^2}{n+rd}\Bigl[n + rd - \operatorname{tr}\bigl((C_u + r^2C_d)(C_u+rC_d)^{-1}\bigr)\Bigr] \qquad \text{(9)}$$

$$\mathbb{E}\left[L_{\text{test}} \mid X_{\text{in}}\right] = \|\beta_{\text{out}}\|_2^2\Bigl[1 + \operatorname{tr}\bigl((C_u+rC_d)^{-1}(C_u+r^2C_d)(C_u+rC_d)^{-1}\bigr)\Bigr] \qquad \text{(10)}$$

The $\|\beta_{\text{out}}\|_2^2$ prefactor is the irreducible part: signal the learner structurally cannot see. The bracketed term in (10) is the aliasing penalty, and notice that both $C_u + rC_d$ and $C_u + r^2C_d$ appear in it. When those two matrices disagree — which is exactly when $r$ is large and $C_d$ is non-negligible — the penalty inflates.

### 7.4 The three regimes, again

Equation (10) is non-monotonic in $r$ at fixed $(n, d, m)$, and the mechanism mirrors Section 4:

| $r$ | What happens in (10) | Test loss |
|----|----|----|
| Small | Duplicated rows carry little extra weight; predictor barely moves | Near baseline |
| Intermediate | Duplicated block is influential but too large to be absorbed | **Peak** |
| Large | The $r^2$ block saturates the rank of $C_u + r^2C_d$ relative to $C_u + rC_d$ | Recovers toward the memorize-and-isolate point |

### 7.5 The dictionary between toy and reality

| Linear model                | Language model                      |
|-----------------------------|-------------------------------------|
| $m$, observed dimension   | $N$, model capacity               |
| $d$, duplicated pool size | $D_r$, unique repeated tokens     |
| $r$, duplication count    | $R$, repeat count                 |
| $n$, unique sample count  | Training duration ($\mathrm{OT}$) |

Two predictions come out of the theory and both match the transformer sweeps:

1.  **Increasing $m$ shifts the peak to larger $d$** — the analogue of $D_r^{\text{peak}}$ growing with $N$ in equation (4).
2.  **Weak dependence on $n$** — the analogue of the near-$\mathrm{OT}$-independence in Section 5.

### 7.6 Simulations

Direct OLS Monte Carlo matches the closed forms to numerical precision. Excess test loss over a clean baseline is non-monotonic in $d$ at fixed $(m, r)$, and the peak location moves to larger $d$ as $m$ grows — reproduced in both theory and simulation on a grid of $m$ from 24 to 384.

The authors also build a **sample-efficiency analogue of CEG**: with duplicated rows again at 10% of the budget, $\mathrm{SE} = N^\star_{\text{clean}}/N_{\text{actual}}$ is the clean sample budget needed to match the duplicated run's loss. SE drops sharply at intermediate $r$ and partially recovers at extreme $r$ — the same shape as the CEG curves from the language models, produced by a linear regression with no attention, no depth, and no optimizer dynamics.

That's the argument: **the peak is a generic statistical feature of duplicated samples in a misspecified learner**, not a transformer pathology. The authors are appropriately modest about how far to push it — it's an explanatory toy model, not a quantitative model of pretraining.

**Check yourself.** In the toy model the labels are *noiseless*. Where, then, does the "noise" that duplication amplifies actually come from? (If you can answer this, you've understood misspecification.)

------------------------------------------------------------------------

## 8. What to do with this if you're training a model

- **Reporting a duplicate *fraction* is not enough.** Two corpora with identical 10% duplication can differ by a large factor in wasted compute. Report the distribution of repeat counts, not just the total.
- **Audit the middle of the repeat distribution, not the tail.** Instinct says to hunt for the document that appears 10,000 times. The instinct is wrong. The documents appearing a few hundred times, collectively spanning a few million tokens, are the expensive ones at 344M scale — and the worst repeat count *drops* as your model grows.
- **Don't trust the loss plot.** A 2–3% eval-loss regression looks like nothing and costs a third of your budget. Convert to compute-equivalent units before deciding whether to care.
- **Overtraining won't save you.** Training longer shifts the level of damage but not which structure is worst.
- **Don't extrapolate equation (4) to frontier scale.** The authors say so themselves; Section 5 shows where it breaks.

------------------------------------------------------------------------

## 9. Critical reading

Bring these to the seminar. Several are the authors' own admissions; two are not.

**Acknowledged by the authors:**

1.  **Scale.** The largest model is 344M. Everything is one architecture family, one tokenizer, one corpus, one repeated-token fraction.
2.  **No seed variance.** Each $(N, \mathrm{OT}, R)$ cell is a single run. Peak prominences of 0.7–2.7% are being asserted without any measurement of run-to-run noise. The robustness argument is consistency across sweeps, not per-cell significance. This is the weakest link in the empirical case and it's worth pressing on.
3.  **Six points, three parameters.** The Chinchilla fit is at the boundary of identifiability, and every CEG number inherits that.
4.  **Incomplete grid.** 344M has only $\mathrm{OT}=1$. The claim that damage grows with $N$ rests on one point at the largest scale.

**Two that the paper does not raise:**

5.  **The embedding problem, and it's a real one.** Look again at the config table. The 34M model has 4.9M non-embedding parameters — **85% of it is embedding tables**. The 344M model is 49% embeddings. But $C = 6NT$ uses *total* $N$, and the input embedding is a lookup, not a matmul: it contributes essentially zero FLOPs per token. So the reported compute overstates real compute by a factor of roughly $N_{\text{total}}/(N_{\text{total}} - N_{\text{input-emb}})$ — about **1.75× at 34M but only 1.34× at 344M**.

    That is a systematic, monotone tilt across exactly the six points the no-repetition scaling law is fit to. It will bias $\gamma$, and $\gamma$ is the exponent that turns loss into compute. Since the headline 33% depends on $\gamma$ through $1/\gamma \approx 3.15$, this is not a rounding concern. Worth asking: does the qualitative result survive refitting on non-embedding FLOPs?

6.  **Exact replay is the easiest case.** The paper studies verbatim document-level duplication. That is precisely the kind of duplication that exact-hash dedup already removes. What actually survives a modern pipeline is near-duplicates, paraphrases, and templated boilerplate. The paper is careful to frame its setting as a controlled case, and the authors elsewhere argue duplication is itself scale-dependent — but the bridge from "exact replay costs 33%" to "your real corpus costs X" is not built here, and it's the bridge a practitioner needs.

**A smaller one:** the main text and Appendix F describe the peak fit with clashing notation — the main text fits a Gaussian to eval loss with $R^{\text{peak}} = 10^\mu$, the appendix fits one to *fractional* loss increase with $R^{\text{peak}} = \mu$ directly. Same idea, $\mu$ living in different spaces. Not an error, but check which one you're reading before reproducing.

------------------------------------------------------------------------

## 10. Glossary

| Term | Meaning |
|----|----|
| **Repeated-token fraction $f$** | Share of training tokens drawn from the repeated pool. Fixed at 0.1 throughout. |
| **Repeat count $R$** | Times each repeated document is replayed. |
| **Repeated-pool size $D_r$** | Unique tokens in the repeated pool. Tied to $R$ by $D_r \approx fT/R$. |
| **Repetition structure** | The pair $(R, D_r)$. The paper's core object: what varies at fixed budget. |
| **Overtraining multiplier $\mathrm{OT}$** | $\mathrm{OT}=1$ means 20 tokens/parameter. $\mathrm{OT}>1$ is overtraining. |
| **CEG** | Compute-Equivalent Gain: clean compute needed to reach loss $L$, over compute actually spent. |
| **CEL** | Compute-Equivalent Loss, $1 - \mathrm{CEG}$. The fraction of budget wasted. |
| **Irreducible floor $E$** | Asymptote of the fitted scaling law. Fitted at 2.365 nats here. |
| **Aliasing term $a_r$** | In the toy model, the error from fitting unobserved signal through observed coordinates. |
| **Memorize-and-isolate** | Large-$R$ regime where a tiny pool is memorized into a quarantined slice of capacity, limiting damage. |

------------------------------------------------------------------------

## 11. Review questions

1.  A colleague says "we repeat 10% of our tokens, same as the paper, so we should expect about 33% compute loss." Give two distinct reasons this inference is wrong.

2.  Derive equation (2) from $f = 0.1$, $T = 20\,\mathrm{OT}N$, and $fT \approx RD_r$. Then compute $D_r$ for a 153M model at $\mathrm{OT}=2$ with $R = 500$.

3.  Using $\mathrm{CEG} = ((L_{\text{base}}-E)/(L_{\text{rep}}-E))^{1/\gamma}$ with $E=2.365$ and $\gamma=0.317$: a 93M model has a clean loss of 3.66. What repeated-run loss corresponds to $\mathrm{CEL} = 0.21$? Express the gap both in nats and as a percentage of total loss.

4.  Explain, without using the phrase "scaling law," why the same percentage loss regression costs more compute at 344M than at 34M.

5.  In the toy model, show that duplicating one row $r$ times contributes $r v^\top v$ to $X_{\text{in}}^\top X_{\text{in}}$ but $r^2 v^\top v$ to $X_{\text{in}}^\top \Sigma_r X_{\text{in}}$. Why does this asymmetry not arise for $r$ genuinely independent samples?

6.  The eval split is held out before repeated pools are built. What alternative explanation for the observed damage does this design choice rule out?

7.  $R^{\text{peak}} \propto N^{-0.96}$ and $D_r^{\text{peak}} \propto N^{1.84}$ are fit separately. Use $D_r = 2\,\mathrm{OT}N/R$ to show these exponents are mutually inconsistent, and say which regime the inconsistency matters in.

8.  You're given a corpus audit showing repeat counts of 2, 8, 40, 200, and 6000 across five document clusters. You're training a 344M model. Which cluster do you investigate first, and what would change your answer if you were training a 34M model?

**Short answers**

1.  $(i)$ CEL depends on *structure*, not just fraction — their $R$ distribution may be nowhere near the peak. (ii) 0.33 is specific to 344M at $\mathrm{OT}=1$ against this fitted curve; a different model size sits at a different distance from $E$ and gets a different amplifier. Also acceptable: their duplication is unlikely to be exact document replay.

2.  $fT = 0.1 \times 20\,\mathrm{OT}N = 2\,\mathrm{OT}N$; divide by $R$. For $N=1.53\times10^8$, $\mathrm{OT}=2$, $R=500$: $D_r = 2(2)(1.53\times10^8)/500 \approx 1.2\times10^6$ tokens.

3.  $L_{\text{base}}-E = 1.293$. $L_{\text{rep}}-E = 1.293 \times 0.79^{-0.317} \approx 1.293 \times 1.077 \approx 1.393$. So $L_{\text{rep}} \approx 3.758$ — a gap of about 0.10 nats, roughly **2.7%** of total loss, for a 21% compute loss.

4.  A better model's loss is mostly floor. A 2% move in total loss is therefore a much larger move in the part of the loss that compute can actually buy down — and the compute–loss curve is shallow, so a large move in that part corresponds to an enormous move in compute.

5.  See Section 7.2. For independent samples the hidden-feature realizations differ per row, so the cross terms have mean zero and the covariance stays diagonal — no all-ones block, no $r^2$.

6.  Benchmark/test-set contamination. Any damage must come from a distorted training distribution, not from the model having seen eval data.

7.  $D_r \propto N/R \propto N \cdot N^{0.96} = N^{1.96}$, versus the fitted 1.84. In-range the two fits agree closely (both give ~$5\times10^4$ at 34M and ~$4\times10^6$ at 344M); the discrepancy blows up under extrapolation, which is exactly where the authors warn against using it.

8.  At 344M, $R^{\text{peak}} \approx 155$ — investigate the 200 cluster. At 34M, $R^{\text{peak}} \approx 1400$ — the 6000 cluster becomes the more likely offender and 200 looks comparatively safe.

------------------------------------------------------------------------

## 12. Further reading

**Read first — the direct ancestor.** Hernandez et al. (2022), *Scaling Laws and Interpretability of Learning from Repeated Data*, arXiv:2205.10487. The non-monotonic result, the effective-parameter framing this paper replaces, and the induction-head connection.

**The measurement backbone.** Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla). Source of the $(N,T)$ allocation rule and the fitted form used throughout. Kaplan et al. (2020), arXiv:2001.08361 — the original power-law scaling paper. Besiroglu et al. (2024), arXiv:2404.10102 and Porian et al. (2024) — replication and reconciliation of Chinchilla fits. Directly relevant to Section 6's identifiability caveat.

**The complementary regime.** Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*. The uniform-repetition case and the ~4-epochs-are-free result. Xue et al. (2023), *To Repeat or Not To Repeat*.

**Deduplication and memorization.** Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*. Carlini et al. (2023), *Quantifying Memorization Across Neural Language Models*. Abbas et al. (2023), *SemDeDup*, arXiv:2303.09540 — semantic dedup, the regime this paper's exact-replay setting doesn't cover.

**The statistical machinery behind Section 7.** Belkin et al. (2019), PNAS — double descent. Bartlett et al. (2020), PNAS — benign overfitting in linear regression. Hastie et al. (2022), *Annals of Statistics* — ridgeless least-squares interpolation.

**Data.** Penedo et al. (2024), *The FineWeb Datasets* — the corpus used here.
