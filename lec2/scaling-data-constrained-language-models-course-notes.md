# Scaling Data-Constrained Language Models

**Paper:** Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel (Hugging Face / Harvard / University of Turku, May 2023; NeurIPS 2023) — arXiv:2305.16264 **Format:** self-contained course-note chapter **Prerequisites:** the Transformer (Lecture 1) and GPT-2 (Lecture 2) at the level of "what is a parameter, what is a token." Kaplan et al. scaling laws (Lecture 3) is the direct prerequisite: power-law fits, log-log plots, the $\text{FLOPs} \approx 6ND$ accounting, and what "compute-optimal" means. You also need: cross-entropy test loss, the word *epoch*, the sum of a geometric series, and the Taylor expansion $e^x \approx 1+x$.

------------------------------------------------------------------------

## Chapter 0. The one-sentence version

Chinchilla tells you how to split a compute budget between model size and tokens **on the assumption that you can always get more tokens**. This paper asks what happens when you can't — and finds that reusing the same corpus for **up to about 4 epochs costs you almost nothing**, that meaningful gains continue to roughly **16 epochs**, and that by **40 epochs** repeating is worthless.

It backs this with a modified scaling law that reduces exactly to Chinchilla at one epoch, fitted on 400+ trained models. The law's most actionable consequence is counterintuitive and is where most of the lecture discussion should go:

> When data runs out, **extra parameters go stale faster than extra epochs do**. So spend surplus compute on more passes over your data before you spend it on a bigger model — the opposite of what the intuition "repeated data is worth less, so buy capacity instead" would tell you.

------------------------------------------------------------------------

## Chapter 1. The world before this paper

### 1.1 Two scaling laws, one correction

From Lecture 3 you know Kaplan et al.'s prescription: a 10× increase in compute should buy roughly a 5.5× larger model and only a 1.8× increase in tokens. Model size was the thing to buy. This produced a generation of enormous, under-trained models — the 530B-parameter MT-NLG was trained on just 270B tokens.

Chinchilla (Hoffmann et al., 2022) overturned that. Its headline demonstration: a **70B-parameter model trained on 4× more data beat the 280B-parameter Gopher** at similar compute. Its prescription was that parameters and tokens should scale *in equal proportion* — roughly 20 tokens per parameter.

### 1.2 The uncomfortable extrapolation

Chinchilla's rule is a demand for data, and the demand grows without bound. Extrapolate it to a 530B model and you need about **11 trillion tokens — over 30 terabytes of text**. Villalobos et al. (2022) estimated that high-quality English text on the internet would be exhausted around 2024 at that rate.

And English is the easy case. The paper points out, in a footnote that is worth putting on a slide, that **100 million tokens** is a realistic post-filtering corpus size for languages like Basque, Punjabi, or Slovenian. For most of the world's languages, the data wall is not a 2024 problem — it is the present tense.

So: what do you do when you run out?

### 1.3 The one-epoch dogma

The obvious answer — go around the data again — was, at the time, something close to forbidden. Nearly all large models were trained for a single epoch. One influential paper was titled "One epoch is all you need." Hernandez et al. (2022) found that up-sampling just 0.1% of the training data 100× **significantly degraded** performance, which hardened the rule into folklore.

There was exactly one prominent counterexample: **Galactica** trained for 4.25 epochs and saw loss keep falling. But Galactica never trained a single-epoch control at the same compute, so it couldn't answer the question anyone actually cared about: *how much performance am I giving up by repeating instead of collecting more data?*

Note the difference in what is being repeated, because it is the crux of why this paper reaches a different conclusion than Hernandez et al.:

|                  | Hernandez et al. (2022)    | This paper                 |
|------------------|----------------------------|----------------------------|
| What is repeated | a small subset, up-sampled | the **entire** corpus      |
| Rest of the data | seen once                  | there is no rest           |
| Effect found     | sharp degradation          | negligible up to ~4 epochs |

Up-sampling a sliver of your data creates a distribution skew. Looping over all of it does not. These are different interventions and it is a mistake — a common one in seminar discussion — to treat the earlier result as contradicted.

### 1.4 The actual question

Fix a compute budget $C$ and a **unique data budget** $D_C$. Allocate $C$ to model size $N$ and total processed tokens $D$. Since $D$ may now exceed $D_C$, some of those tokens are repeats.

$$\underset{N,D}{\arg\min}\; L(N,D) \quad \text{s.t.}\quad \mathrm{FLOPs}(N,D) = C,\quad U_D \le D_C$$

Everything in this paper is an answer to two questions about that constrained problem, borrowed from the scaling-law literature:

- **Allocation** — given the budget, what is the best split between $N$ and $D$?
- **Return** — what is an extra FLOP actually worth once the data is being recycled?

------------------------------------------------------------------------

## Chapter 2. The Chinchilla law, in the form you need it here

You cannot follow this paper without holding Chinchilla's parametric fit in your head, because the new law is that formula with two substitutions.

$$L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E$$

Three terms, and each one means something:

- $E$ — **irreducible loss**. The entropy of the text itself. No model, however large, trained on however much data, gets below this. In the paper's C4 fit, $E = 1.87$ nats.
- $A/N^{\alpha}$ — the penalty for having a **finite model**. Shrinks as you add parameters, never reaches zero.
- $B/D^{\beta}$ — the penalty for having seen **finite data**. Shrinks as you process more tokens.

$\{A, \alpha, B, \beta, E\}$ are fitted, not derived. The fit the paper uses for C4 (their Appendix B, reconstructed from Chinchilla's own C4 runs) is:

$$L(N,D) = 1.87 + \frac{521}{N^{0.353}} + \frac{1488}{D^{0.353}}$$

Note $\alpha = \beta$. That equality *is* the Chinchilla conclusion: it forces the optimal allocation to split a compute increase evenly between $N$ and $D$. Specifically,

$$N_{\text{opt}}(C) = G\left(\tfrac{C}{6}\right)^{a},\qquad D_{\text{opt}}(C) = G^{-1}\left(\tfrac{C}{6}\right)^{b},\qquad G = \left(\tfrac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\quad a = \tfrac{\beta}{\alpha+\beta},\quad b = \tfrac{\alpha}{\alpha+\beta}$$

When $\alpha = \beta$, $a = b = 0.5$: multiply compute by 100, multiply both $N$ and $D$ by 10.

**Where the 6 comes from.** The cost model throughout is $\text{FLOPs}(N,D) \approx 6ND$, inherited from Kaplan et al. The standard accounting: a forward pass costs about two FLOPs per parameter per token (one multiply, one add), and the backward pass costs about twice the forward pass, giving $2 + 4 = 6$. The paper cites this rather than deriving it, and the constant doesn't matter much — what matters is that **cost is linear in both $N$ and $D$**, so a parameter and a token are interchangeable currency. That is exactly what makes "should I buy parameters or epochs?" a well-posed question.

**Sanity check that the fit is real.** Plug the Gopher compute budget of $5.76 \times 10^{23}$ FLOPs into the formulas above and you get $N_{\text{opt}} = 70.0$B parameters and $D_{\text{opt}} = 1.37$T tokens. Chinchilla's own IsoFLOP curves on C4 predicted 73B and 1.3T. The reconstruction lands within a few percent, which is why we can trust the coefficients for everything that follows.

**The assumption that breaks.** Chinchilla's fit was made entirely from single-epoch runs. $D$ there means "tokens processed," and every token processed was fresh. Nothing in the formula knows what a repeat *is*, so there is no reason to expect it to extrapolate into the multi-epoch regime — and Chapter 7 shows it doesn't.

> **Check yourself.** If you double compute under Chinchilla, the model gets $\sqrt{2}\times$ bigger and sees $\sqrt{2}\times$ more tokens. Which of the three terms in $L$ does each of those changes touch, and why does the irreducible term make the *percentage* improvement in loss shrink as models get better?

------------------------------------------------------------------------

## Chapter 3. Bookkeeping: splitting $D$ and $N$ in two

Before any new mathematics, the paper does an accounting move. It splits each of $D$ and $N$ into a "unique" part and a "repeat" part.

### 3.1 Splitting the data

Given a total token count $D$ and a unique-data budget $D_C$:

$$U_D = \min(D_C, D) \qquad\qquad R_D = \frac{D}{U_D} - 1$$

$U_D$ is how many distinct tokens you actually have. $R_D$ is the number of **repetitions**, which is epochs minus one. Train one epoch and $R_D = 0$. Train four epochs and $R_D = 3$. Keep that offset straight — it is the single most common source of confusion when reading the plots, where the axis is usually labelled in *epochs* while the equations are in *repeats*.

### 3.2 Splitting the parameters

This one is less obvious and repays slow reading.

$$U_N = \min(N_{\text{opt}}(U_D), N) \qquad\qquad R_N = \frac{N}{U_N} - 1$$

To compute $U_N$, you ask Chinchilla: *if I had exactly $U_D$ unique tokens and trained for one epoch, what model size would be compute-optimal?* That size is $U_N$ — the parameters your unique data can "support." Anything beyond it is **excess**, counted by $R_N$.

With the fitted C4 coefficients this relationship collapses to a memorable constant:

$$U_N = 0.051 \cdot U_D$$

That is roughly **one parameter per 20 tokens** — Chinchilla's ratio, reappearing as the definition of what counts as excess capacity.

*Concrete instance.* You have $U_D = 100$M unique tokens. Then $U_N \approx 5$–7M parameters is "what the data supports." Train a 212M-parameter model on it and $R_N \approx 30$: you have thirty times more capacity than your data can justify.

### 3.3 Why symmetry

The reason to split $N$ as well as $D$ is a defect in Chinchilla the authors point out directly. Under $L = A/N^\alpha + B/D^\beta + E$, growing a model from 1B to 10B parameters produces the *same absolute loss reduction* whether your dataset is a billion tokens or a single token. The second case is nonsense — nine billion extra parameters cannot extract anything from one token that the first billion missed. Chinchilla never had to confront this because it only ever looked at compute-optimal configurations, where $N$ and $D$ are matched by construction. Once you allow either to run far ahead of the other, the formula needs a way for surplus to stop mattering.

Both problems have the same shape, so they get the same fix.

------------------------------------------------------------------------

## Chapter 4. The effective-data formula

This is the mechanical heart of the paper. Everything else is experiment.

### 4.1 The modelling assumption

Replace $D$ in Chinchilla's formula with **effective data** $D'$: the number of *fresh* tokens that would have been worth as much as the repeated tokens you actually processed. We want $D' \le D$, with equality at one epoch.

The assumption, stated plainly: **each time the model sees a token, it extracts a fixed fraction of whatever information is still left in it.** Let $\delta \in [0,1]$ be the fraction *lost* per repetition, so a token's $k$-th viewing is worth $(1-\delta)^k$ of its first.

Two sanity poles: $\delta = 0$ means repeats are as good as fresh data; $\delta = 1$ means repeats are worthless. Reality is in between, and the whole paper is an attempt to measure where.

This is an *assumption*, not a derivation from learning theory. Its justification is that the resulting curve fits 400+ training runs — check Chapter 5.5 before accepting it as physics.

### 4.2 Sum the series

If value decays by a constant factor per pass, total value is a geometric series:

$$D' = U + (1-\delta)U + (1-\delta)^2 U + \cdots + (1-\delta)^{R_D} U$$

Using $S = a(1-r^n)/(1-r)$ with $r = (1-\delta)$:

$$D' = U + (1-\delta)U\frac{1 - (1-\delta)^{R_D}}{\delta}$$

Already usable. But $\delta$ is an awkward thing to report — "tokens lose 6.5% of their value per repetition" doesn't tell you when to stop.

### 4.3 Reparameterize into something interpretable

As $R_D \to \infty$, the trailing factor $\to 1$ and the whole expression approaches $U + \frac{(1-\delta)}{\delta}U$. Define

$$R_D^{\ast} = \frac{1-\delta}{\delta}$$

so effective data **plateaus at $U + R_D^{\ast} \cdot U$** no matter how many times you go around. This is the most important structural fact in the paper:

> There is a hard ceiling. A corpus of $U$ tokens contains at most $U(1 + R_D^{\ast})$ fresh-tokens-worth of signal, and no amount of compute extracts more.

### 4.4 Clean it up with the exponential

For small $\delta$, two approximations: $1/R_D^{\ast} = \delta/(1-\delta) \approx \delta$, and from $e^x \approx 1 + x$ with $x = -\delta$, $(1-\delta) \approx e^{-\delta} \approx e^{-1/R_D^{\ast}}$. Substituting gives the form used everywhere in the paper:

$$\boxed{D' = U_{D} + U_{D} R_{D}^{\ast}\left(1 - e^{-R_{D}/R_{D}^{\ast}}\right)}$$

Read it in three regimes:

| Regime | Behaviour | Interpretation |
|----|----|----|
| $R_D = 0$ | $D' = U_D = D$ | reduces **exactly** to Chinchilla |
| $R_D \ll R_D^{\ast}$ | $D' \approx U_D(1+R_D) = D$ | repeats ≈ fresh data |
| $R_D \gg R_D^{\ast}$ | $D' \to U_D(1 + R_D^{\ast})$ | ceiling; extra epochs buy nothing |

$R_D^{\ast}$ is a **half-life for repetition**. At exactly $R_D = R_D^{\ast}$, the repeated tokens have retained $`1 - 1/e \approx 63\%`$ of their average value.

That the formula collapses to Chinchilla at $R_D = 0$ is not a coincidence — it is a design requirement. A new law that disagreed with the old one in the regime the old one was fitted on would be a worse law, not a better one.

### 4.5 A worked example you can do by hand

The paper's own, and worth reproducing on the board. Take $\delta = 0.25$ (repeats retain 75% of value), $U = 1$ unit of data, and train **5 epochs** ($R_D = 4$).

Exact geometric sum:

$$D' = 1 + 0.75 \cdot \frac{1 - 0.75^4}{0.25} = 1 + 3(1 - 0.316) = 3.05$$

So five passes over one unit of data bought you the value of **3.05 units**. Four of the five passes were repeats and together they were worth about two fresh units.

Now the approximation. $R_D^{\ast} = (1-\delta)/\delta = 3$:

$$D' = 1 + 3\left(1 - e^{-4/3}\right) = 3.21$$

3.21 versus 3.05 — a 5% discrepancy, which looks bad until you remember $D'$ enters the loss raised to $\beta = 0.353$. The resulting difference in the loss term is $`(3.21/3.05)^{0.353} - 1 = \mathbf{1.8\%}`$. The exponent flattens the error, which is why the authors accept the approximation in exchange for an interpretable $R_D^{\ast}$.

Push it further: at $R_D = 100$, $D' = 1 + 3(1 - e^{-33.3}) = 3.99$. The ceiling $U(1+R_D^{\ast}) = 4$ is essentially reached. **A hundred epochs on this corpus is worth four fresh corpora — and a thousand epochs is still worth four.**

### 4.6 The table to actually remember

Using the paper's fitted $R_D^{\ast} = 15.4$ (Chapter 5), here is what each epoch buys. "Marginal value" is what the $n$-th pass is worth relative to a fresh pass, $e^{-R_D/R_D^{\ast}}$; "efficiency" is cumulative effective data divided by tokens processed.

| Epoch $n$ | Marginal value of this pass | Cumulative effective corpora | Efficiency |
|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 100% |
| 2 | 0.94 | 1.97 | 98% |
| 3 | 0.88 | 2.88 | 96% |
| 4 | 0.82 | 3.73 | 93% |
| 5 | 0.77 | 4.52 | 90% |
| 8 | 0.64 | 6.62 | 83% |
| 16 | 0.38 | 10.58 | 66% |
| 40 | 0.08 | 15.17 | 38% |
| 100 | 0.002 | 16.36 | 16% |
| $\infty$ | 0 | 16.39 | 0% |

Every headline claim in the paper is visible in this column of numbers: 4 epochs is 93% efficient (≈ free), 16 epochs is where the marginal pass drops below 40%, and 40 epochs is where you are burning three FLOPs for every one that does work.

> **Check yourself.**
>
> 1.  Your corpus is 10B tokens and $R_D^{\ast} = 15.4$. What is the largest effective dataset you could ever reach, and how much compute would be wasted getting to 90% of it?
> 2.  Suppose you measured $\delta = 0.5$ instead. What is $R_D^{\ast}$, and how does the guidance change?

------------------------------------------------------------------------

## Chapter 5. Effective parameters, and the full law

### 5.1 The symmetric substitution

Excess parameters get exactly the same treatment, with its own learned constant:

$$N' = U_N + U_N R_N^{\ast}\left(1 - e^{-R_N/R_N^{\ast}}\right)$$

The story is the same one: beyond the size your data supports, each additional slab of capacity learns features the previous slab already learned, and its contribution decays exponentially. Ceiling: $N'$ never exceeds $U_N(1 + R_N^{\ast})$.

Put both substitutions into Chinchilla and you have the paper's law:

$$L(U_N, U_D, R_N, R_D) = \frac{A}{\left(U_N + U_N R_N^{\ast}\left(1 - e^{-R_N/R_N^{\ast}}\right)\right)^{\alpha}} + \frac{B}{\left(U_D + U_D R_D^{\ast}\left(1 - e^{-R_D/R_D^{\ast}}\right)\right)^{\beta}} + E$$

If you set $R_N^{\ast} = R_D^{\ast} = \infty$, both decay terms vanish and you are back at Chinchilla exactly. It is a strict generalization, which is the right shape for a scaling law to have.

### 5.2 How the constants were fitted

Worth knowing, because the caveats live here.

- $\alpha, \beta, A, B, E$ are **fixed** to the C4 values from Chapter 2. Only $R_N^{\ast}$ and $R_D^{\ast}$ are learned — two free parameters, not seven.
- Objective: Huber loss on the log-sum-exp form of the equation (following Chinchilla's methodology), minimized with LBFGS from a grid of initializations.
- **182 runs**, spanning 7M to 9B parameters and 1 to 500 epochs.
- **Outliers were removed** — specifically, runs where excess parameters or excess epochs made loss *worse*. The functional form cannot represent that (it only ever plateaus), so such runs were excluded from the fit. Hold onto this; Chapter 11 returns to it.

### 5.3 The two numbers

$$R_N^{\ast} = 5.31 \qquad\qquad R_D^{\ast} = 15.39$$

Interpretation, in the units that matter:

- **Data:** you can repeat roughly 15 times before hitting sharply diminishing returns. A corpus is worth at most ~16.4× its size.
- **Parameters:** you can exceed the data-supported model size by roughly 5.3× before extra capacity stops helping. Ceiling ~6.3× $U_N$.
- Recovered decay rates: $\delta_D \approx 1/15.39 \approx 0.065$ and $\delta_N \approx 1/5.31 \approx 0.19$, both comfortably small enough for the approximation in §4.4 to hold.

### 5.4 The inequality that drives everything

$$R_D^{\ast} > R_N^{\ast}$$

Excess parameters decay **about three times faster** than repeated data. This single inequality produces the paper's central practical recommendation and its disagreement with Chinchilla:

|  | Chinchilla | Data-constrained |
|----|----|----|
| Surplus compute goes to | $N$ and $D$ equally | epochs **faster** than parameters |
| Holds when | data is unlimited | unique data is capped |
| Agree when | — | $R_D = R_N = 0$ (single epoch) |

The intuition to *avoid* is the natural one: "repeated data is degraded, so I should compensate by buying a bigger model." The finding is the reverse. **Parameters learning from repeated data are worth even less than the repeated data itself.** Once you have run out of new text, the cheaper move is another pass, not a wider model — and a smaller model is cheaper at inference time too.

### 5.5 Does the decay term earn its keep?

The paper reports fits of several variants against the same 182 runs. This ablation is the evidence that the exponential decay isn't decoration:

| Variant | $R_D^{\ast}$ | $R_N^{\ast}$ | $R^2$ |
|----|---:|---:|---:|
| No decay (plain Chinchilla on repeated data) | — | — | 0.445 |
| Decay parameters only | — | 713.0 | 0.449 |
| Decay data only | 2.92 | — | 0.735 |
| **Decay both (Eq. 14)** | **15.39** | **5.31** | **0.772** |
| Exact geometric form, both | — | — | 0.799 |

Three readings. First, plain Chinchilla explains less than half the variance once data is repeated — it really is the wrong model here. Second, decaying the data term is where most of the gain comes from, but decaying parameters adds a further six points. Third, the exact geometric form (§4.2) fits marginally *better* than the approximation the paper adopts; they choose the approximation anyway, for interpretability. Being explicit about paying ~2.7 points of $R^2$ for a formula humans can reason about is good practice and worth pointing at in discussion.

> **Check yourself.** The "decay parameters only" row learned $R_N^{\ast} = 713$. What does a value that large mean the fit is saying, and why is it a symptom of a misspecified model rather than a discovery?

------------------------------------------------------------------------

## Chapter 6. The experiments

### 6.1 Setup

- **Architecture:** GPT-2, with the GPT-2 tokenizer — chosen for comparability, not performance. Models up to 8.7B parameters, runs up to 900B total tokens, more than **400 models** trained, up to 1500 epochs.
- **Data:** subsets of C4.
- **Schedule:** cosine learning rate decaying 10× over each run, following Chinchilla. Schedule choice materially changes fitted scaling coefficients, so matching Chinchilla here is deliberate.
- **No early stopping** — on purpose, so overfitting from repetition could be observed rather than hidden.

### 6.2 One design detail worth the slide: nested subsets

When comparing a run using 44B unique tokens to one using 178B, the smaller budget is always a **subset** of the larger. So the runs differ in *how many times data is seen*, not in *which data is seen*. Without this, every comparison would be confounded by data composition. Data is shuffled between epochs, and the entire corpus is repeated — never a subset up-sampled (the contrast with Hernandez et al. from §1.3).

### 6.3 Test loss, not training loss

Chinchilla fitted on training loss. This paper uses held-out test loss, and must: when you repeat data, training loss becomes actively misleading. Their Appendix H figure shows models trained on **fewer** unique tokens achieving **better** training loss — because they are memorizing. Any metric that rewards more repetition by construction cannot be used to study repetition. A short but instructive methodological point: a metric that is fine in one regime can invert in another.

### 6.4 The three protocols

| Protocol | What's fixed | What varies | Question answered |
|----|----|----|----|
| Fixed Unique Data (§7) | $D_C \in \{100\text{M}, 400\text{M}, 1.5\text{B}\}$ | parameters, epochs | **Allocation** |
| Fixed FLOPs (§8) | compute budget | $D_C$ (hence epochs) | **Return** |
| Parametric Fit | — | all runs | validates the law |

------------------------------------------------------------------------

## Chapter 7. Result I — Allocation: how to spend a budget

Take $D_C = 100$M unique tokens. Chinchilla's one-epoch compute-optimal model for that data is about **7M parameters**. That is the starting point; the experiment asks what happens as you spend more compute than is "optimal."

**The result is large.** Across 93 models on this budget, the best loss is achieved at roughly **20–60× more parameters and 20–60× more epochs** than compute-optimal — around **7000× more FLOPs** — and it cuts loss by more than **50%** (the contour plot spans 8.10 down to 3.72).

The lesson to draw is not "compute-optimal is wrong." It is that **compute-optimal and loss-optimal are different objectives**, and they diverge violently when data is the binding constraint. A one-epoch model at a fixed data budget is leaving most of the extractable signal in the corpus untouched. If you have 100M tokens and a cluster, the 7M-parameter model is the wrong model to train — even though it is exactly what Chinchilla prescribes.

**The frontier bends.** In the single-epoch, near-compute-optimal corner, the two laws' efficient frontiers overlap — as they must, since the new law reduces to the old there. As epochs increase, the data-constrained frontier peels away, allocating most additional compute to **epochs** rather than parameters, exactly as $R_D^{\ast} > R_N^{\ast}$ demands. The same pattern holds at all three data budgets (100M, 400M, 1.5B).

**Confirmation at scale.** The prediction was tested where it costs real money: $9.3 \times 10^{21}$ FLOPs, 25B unique tokens.

|  | Chinchilla-style allocation | Data-constrained allocation |
|----|----|----|
| Parameters | 8.67B | **6.34B** (27% fewer) |
| Tokens processed | 178B | 242B |
| Epochs | 7.1 | 9.7 |
| Final loss | 2.376 | **2.359** |
| Avg. downstream (19 tasks) | 23.5 | **25.9** |

Smaller model, more passes, same compute, better on both loss and downstream. And it is cheaper to serve afterwards.

**A caveat the authors state plainly.** Push parameters and epochs far enough and loss stops improving and starts *rising* again. The law does not model this — it predicts a plateau, never a decline. See Chapter 11.

> **Check yourself.** The best configuration at 100M tokens uses ~7000× the compute-optimal FLOPs for a ~50% loss reduction. Under what circumstances is that a rational trade, and under what circumstances is it obviously not? Your answer should mention what else you could have spent those FLOPs on.

------------------------------------------------------------------------

## Chapter 8. Result II — Return: what is an extra FLOP worth?

Now hold compute fixed and vary the data budget. Three IsoFLOP settings:

| FLOPs                  | $N$ | $D$ (total) | Unique-token budgets $D_C$ |
|------------------------|-------|---------------|------------------------------|
| $9.3 \times 10^{20}$ | 2.8B  | 55B           | 55B → 1.25B (1 to 44 epochs) |
| $2.1 \times 10^{21}$ | 4.2B  | 84B           | 84B → 1.9B                   |
| $9.3 \times 10^{21}$ | 8.7B  | 178B          | 178B → 4B                    |

Every model in a row processes the same number of tokens and has the same size. The only difference is how many of those tokens are distinct.

**The headline number.** The 8.7B model trained for **4 epochs** on 44B unique tokens finishes with only **0.5% higher validation loss** than the same model trained on 178B unique tokens for one epoch. Three quarters of the data was thrown away and replaced with repetition, at a cost of half a percent.

You can reproduce that from the law. With $U_D = 44$B and $R_D = 3.05$, effective data is $D' \approx 166$B against $D = 178$B processed, and running the full equation predicts a loss increase of about 0.7% — the right order of magnitude, from a two-parameter fit.

**Where it breaks down.** Extrapolating the law out to enormous compute with $D_C$ held fixed reproduces the three-phase shape from the paper's opening figure:

1.  **Up to ~4 epochs** — repeating is nearly free.
2.  **Up to ~16 epochs** ($\approx R_D^{\ast}$) — still worth doing, returns visibly compressing.
3.  **By ~40 epochs** — repeating is worthless; curves flatten and added compute buys nothing.

**A prediction failure the authors report on themselves.** The fit *underestimates* final loss for runs that fail — models trained for 44 epochs where loss rises mid-training. The law is accurate in the regime you would actually operate in and optimistic in the regime you should already have left.

> **Check yourself.** You have 10B unique tokens and enough compute for 200B training tokens. Using the epoch table in §4.6, roughly what fraction of your compute is doing useful work? Would you rather have 20B unique tokens and half the compute?

------------------------------------------------------------------------

## Chapter 9. Case study: Galactica

Galactica is the one real model the paper can second-guess with its own law, and it makes the lesson concrete.

What Galactica did: 106B unique tokens, a **120B-parameter** model, 450B tokens processed — **4.25 epochs**. The team stopped there, citing a small bump in validation loss at the start of the fifth epoch.

What the data-constrained law says they should have done: **40B parameters** trained on **1.35T tokens** — about **12.75 epochs**. Three times smaller, three times more passes, same compute.

Two separate errors, one per question:

- **Allocation.** 120B parameters on 106B unique tokens is heavy on capacity even by Chinchilla's standards, and the reasoning behind it was probably the intuition §5.4 refutes — *repeats are degraded, so buy parameters.* Parameters fed on repeats are the more degraded purchase.
- **Return.** The paper's own epoch-by-epoch loss curves show small spikes when a new epoch begins, which then resolve and continue falling. Galactica's fifth-epoch bump looks like exactly that pattern, so stopping on it likely cost real performance.

Two honest caveats the authors flag: the law was fitted on C4, while Galactica trained on scientific text including code, and scaling coefficients do vary by corpus. The direction of the correction should hold; the precise "40B / 1.35T" should not be taken literally.

------------------------------------------------------------------------

## Chapter 10. When repeating isn't enough

Repetition has a ceiling, so §7 of the paper asks what else you can pour into a fixed data budget. Setup: 4.2B-parameter models, 84B total tokens, evaluated on **19 natural-language tasks** at 0–5 shots (114 scores per model), with scores normalized from each task's random baseline so generative tasks don't dominate the average. Most points are averaged over five seeds.

**Repeating (downstream).** Differences are insignificant up to about **4 epochs** (25% of the budget unique), then performance drops. This independently confirms the loss-based finding on actual tasks — worth noting, because loss and downstream accuracy do not always move together.

**Filling with code.** Replace missing natural-language data with Python from The Stack. Up to **50% code** (42B tokens), performance on *natural-language* tasks is undiminished. Beyond that it falls off quickly. Two tasks jump as soon as any code is added: **bAbI** (reasoning) and **WebNLG** (generation) — the authors' hypothesis is that code teaches long-range state tracking. A telling detail: C4 models score **zero** on bAbI, because C4's creators deliberately stripped anything resembling code; OSCAR models, which get code by accident, score well above random.

**Filtering.** Two common filters, revisited as a way to *free up* data by not applying them:

| Strategy | Tokens surviving | Epochs to fill budget | Downstream effect |
|----|----|----|----|
| Perplexity filter (keep lowest-perplexity 25%) | 44B | ~2 | helps |
| Deduplication (drop 100-char overlaps) | 21B | 4 | no benefit measured |

Deduplication was previously shown to improve perplexity, but that work didn't evaluate downstream; and dedup may still be worth doing for reasons outside this benchmark, such as reducing memorization of training data. The recommendation is conditional: **reserve aggressive filtering for genuinely noisy corpora** (they find it more effective on the noisier OSCAR).

**The practical recipe**, which is the sentence to take into a real project:

> Double your data by mixing in code, then repeat the result for four epochs. That is **8× the training tokens**, and the paper's evidence says it should perform about as well as having had 8× more unique text in the first place.

------------------------------------------------------------------------

## Chapter 11. What the law deliberately does not model

Scaling laws are fitted curves with domains of validity. This one has four boundaries, and a seminar that skips them has misread the paper.

**1. Excess capacity can actually hurt, not plateau.** Empirically, loss rises once parameters or epochs go far enough past optimal. The formula cannot express this — its exponentials decay to zero contribution, never negative. The authors' argument is that plateau is the *right* modelling choice because the decline is preventable: in the limit, "regularize away the excess parameters" just means removing them, and you would stop training anyway once loss flattens. Defensible, but it does mean the law is optimistic exactly where a practitioner might get burned. They also try an alternative formulation that decays the *exponents* $\alpha,\beta$ instead, which does let excess hurt — and find it predicts returns from repeated data noticeably worse, with no principled justification. It was rejected.

**2. Hyperparameters may be the real culprit.** They test whether excess parameters hurt under μP, a principled hyperparameter-transfer scheme. It still hurts — and μP actually gave *higher* test loss than their defaults. Their hypothesis is that the missing ingredient is explicit regularization (dropout and similar), which μP doesn't cover. This remains untested and is a legitimate open question to raise in discussion.

**3. Double descent adds noise to the fit.** Loss on 100M tokens *increases* around 200 epochs and then decreases again — epoch-wise double descent. Since the law assumes loss decreases monotonically in epochs, these runs are noise with respect to the fit, and most were removed. Note the mild circularity: runs that contradict the functional form were excluded from the data used to validate the functional form. The authors are open about this; it is still a limitation.

**4. Everything is C4 and GPT-2.** The coefficients are dataset-specific. They do check robustness — OSCAR reproduces the same qualitative trends, and deduplicated C4 gives the same optimal epoch count (59) as regular C4 — so the *shape* travels. The *numbers* should not be transplanted to a different corpus without refitting, and nothing here speaks to modern architectures, data curricula, or multi-trillion-token corpora.

------------------------------------------------------------------------

## Chapter 12. Why it mattered

The narrow contribution is a scaling law with two extra constants. The broad contribution is a defensible answer to a question that had been answered by folklore.

- **It made "repeat your data" respectable.** Before this, multi-epoch pretraining at scale was something you did apologetically and without a control. After, "up to ~4 epochs is roughly free" became a standard planning assumption, with numbers to cite.
- **It reframed the data wall.** Running out of unique text does not stop scaling; it changes the exchange rate. Compute remains buyable, up to a corpus-dependent ceiling of roughly 16× your unique tokens.
- **It inverted a plausible-sounding heuristic.** "Data is degraded, so buy parameters" is exactly wrong. This is the kind of result scaling laws exist to produce — the arithmetic disagreeing with the intuition, with 400 runs behind the arithmetic.
- **It is reproducible.** 400+ models and the datasets were released (the `huggingface/datablations` repository), which is a large part of why the result propagated rather than being taken on faith.

Where it sits in this course: Lecture 3 (Kaplan) established that loss is predictable from compute. Chinchilla corrected the allocation. This paper removes the assumption that data is free — the last of the three resources to be treated as unbounded. It is also a good template for how to write a scaling paper: generalize an accepted law so it reduces to that law in the old regime, fit the minimum number of new constants, and be explicit about where your curve is wrong.

------------------------------------------------------------------------

## Glossary

| Term | Meaning |
|----|----|
| **Allocation** | How to divide a fixed compute budget between model size and tokens processed. |
| **Return** | How much loss improvement an additional unit of compute buys. |
| **$D_C$** | Unique data budget — the total distinct tokens available. |
| **$U_D$** | Unique tokens actually used, $\min(D_C, D)$. |
| **$R_D$** | Repetitions = epochs − 1. One epoch means $R_D = 0$. |
| **$U_N$** | Parameters the unique data supports: Chinchilla-optimal $N$ for $U_D$. Roughly $0.051 \cdot U_D$ on C4. |
| **$R_N$** | Excess-parameter multiplier, $N/U_N - 1$. |
| **$D'$, $N'$** | Effective data / effective parameters — the fresh-equivalent amounts after decay. |
| **$R_D^{\ast}$** | Learned data half-life, **15.39**. Repeats past ~15 yield sharply diminishing returns; ceiling $U_D(1+R_D^{\ast})$. |
| **$R_N^{\ast}$** | Learned parameter half-life, **5.31**. Smaller than $R_D^{\ast}$ ⇒ prefer epochs over parameters. |
| **$E$** | Irreducible loss — entropy of the data. 1.87 nats on C4. |
| **IsoFLOP** | A set of runs holding total compute constant while varying other factors. |
| **IsoLoss contour** | A curve through (epochs, parameters) configurations achieving equal loss. |
| **Efficient frontier** | The locus of best-achievable loss per compute budget. |
| **Compute-optimal** | Best loss *per FLOP* — not the same as best achievable loss at a fixed data budget. |
| **Epoch-wise double descent** | Loss rising then falling again as epochs increase. |
| **μP** | Maximal Update Parameterization; a hyperparameter-transfer scheme tested here. |

------------------------------------------------------------------------

## Review questions

**Recall**

1.  State the Chinchilla parametric loss and say in one sentence what each of its three terms represents.
2.  Write $U_D$, $R_D$, $U_N$, $R_N$ in terms of $D$, $N$, and $D_C$. How many repetitions does a 6-epoch run have?
3.  Give the fitted values of $R_D^{\ast}$ and $R_N^{\ast}$ and state the ceiling each implies.

**Mechanism**

4.  Derive $D' = U + U R_D^{\ast}(1 - e^{-R_D/R_D^{\ast}})$ from the geometric-series form, naming the two approximations used and the assumption each requires.
5.  Show that at $R_D = 0$ the full law reduces exactly to Chinchilla. Why is that a design requirement rather than a lucky coincidence?
6.  The paper's own example has the approximation giving $D' = 3.21$ where the exact sum gives 3.05 — a 5% gap that becomes 1.8% in the loss. Explain the mechanism that shrinks it, and say what would happen to that gap if $\beta$ were 0.9 instead of 0.353.
7.  Why must $N$ be split into $U_N$ and $R_N$? Construct the absurd case that plain Chinchilla permits.
8.  Explain, using $R_D^{\ast} > R_N^{\ast}$, why surplus compute should go to epochs before parameters. Then state the plausible-sounding intuition this refutes and identify precisely where that intuition goes wrong.

**Application**

9.  You have 2B unique tokens and a budget for 24B training tokens. Using the table in §4.6, estimate your effective data and the fraction of compute doing useful work. Would you rather double your unique data or double your compute?
10. A colleague proposes training a 13B model on a 1B-token corpus because "more capacity compensates for less data." Using $U_N \approx 0.051 U_D$ and $R_N^{\ast}$, estimate the effective parameter count and explain what happens to the other 12-point-something billion.
11. Recompute the Galactica recommendation qualitatively: why does the law move parameters *down* and epochs *up* rather than the reverse, given that repeated data is the degraded resource?

**Critique**

12. The law predicts loss to plateau, but empirically excess parameters and epochs make it rise. Runs exhibiting that rise were removed before fitting. Is this sound methodology? Argue both sides, and state what experiment would settle it.
13. §5.5 shows the exact geometric form fits better ($R^2$ 0.799 vs 0.772) than the approximation the paper adopts. Defend the choice; then argue against it.
14. Everything is fitted on C4 with GPT-2. Which of the paper's conclusions would you expect to survive a change of corpus and architecture, and which are numbers you would refuse to reuse without refitting?

------------------------------------------------------------------------

## Further reading

**Direct prerequisites**

- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*, arXiv:2001.08361 — Lecture 3. Where $6ND$ and the power-law framing come from.
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*, arXiv:2203.15556 — **Chinchilla**. The single most important paper to have read before this one; §3 of these notes is meaningless without it.

**The data-wall framing**

- Villalobos et al. (2022), *Will we run out of data?*, arXiv:2211.04325 — the estimate that motivates the whole enterprise.
- Komatsuzaki (2019), *One epoch is all you need*, arXiv:1906.06669 — the position this paper overturns.

**On repetition and duplication**

- Hernandez et al. (2022), *Scaling Laws and Interpretability of Learning from Repeated Data*, arXiv:2205.10487 — up-sampling a *subset*; contrast carefully with repeating the whole corpus (§1.3).
- Lee et al. (2021), *Deduplicating Training Data Makes Language Models Better*, arXiv:2107.06499 — the deduplication result this paper partially contests on downstream tasks.
- Nakkiran et al. (2021), *Deep double descent*, J. Stat. Mech. — background for the epoch-wise double descent in Appendix D.

**Models referenced**

- Taylor et al. (2022), *Galactica*, arXiv:2211.09085 — the case study in Chapter 9.
- Rae et al. (2021), *Gopher*, arXiv:2112.11446 — the 280B model Chinchilla beat.

**Artifacts**

- `huggingface/datablations` — the 400+ trained models and datasets from this paper.
- Raffel et al. (2020), *T5* (JMLR) — source of the C4 corpus, including the code-stripping decision that explains the bAbI result in Chapter 10.
