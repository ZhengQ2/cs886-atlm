# Training Compute-Optimal Large Language Models (Chinchilla)

**Paper:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Laurent Sifre et al. (DeepMind), *Training Compute-Optimal Large Language Models*, arXiv:2203.15556, March 2022.

**Thesis in one sentence:** For a fixed training compute budget, model size and the number of training tokens should be scaled in *equal* proportion — which means essentially every large language model built before 2022 was far too big for the amount of data it was trained on.

**Prerequisites:** Lecture 1 (the Transformer — you need the architecture to follow the FLOP accounting) and Lecture 3 (Kaplan et al., *Scaling Laws for Neural Language Models* — this paper is a direct rebuttal to it).

**How to read this chapter:** Chapters 1–2 set up the decision problem and the answer the field had already accepted. Chapter 3 is the single methodological point on which the whole disagreement turns; if you only read one chapter, read that one. Chapters 4–6 are the three independent estimation procedures. Chapters 7–9 are the prediction and the model built to test it. Chapters 10–12 are limitations, textual wrinkles, and what happened afterwards.

------------------------------------------------------------------------

## 1. The decision problem

### 1.1 Why this question has a strange shape

Most machine learning questions are answered empirically: try several settings, keep the best. Frontier language model pre-training does not allow that. A single run consumes a datacenter for weeks, so in practice you get **one shot**. The compute budget is also typically fixed *before* the run: you know how many accelerators you have and how long you may keep them.

So the question is not "what is the best model?" but a constrained-allocation question: given a budget you cannot change, what do you spend it on?

There are only two things to spend it on. You can buy **parameters** ($N$) — a wider or deeper network — or you can buy **tokens** ($D$) — more text pushed through it. Every FLOP spent on one is a FLOP not spent on the other.

The paper states this as a constrained optimization:

$$(N_{\mathrm{opt}}(C), D_{\mathrm{opt}}(C)) = \underset{N,D:\,\mathrm{FLOPs}(N,D)=C}{\arg\min}\; L(N,D)$$

Read it left to right: over all pairs $(N, D)$ whose training cost is exactly the budget $C$, find the pair that minimizes final pre-training loss $L$. The output is two functions of the budget, $N_{opt}(C)$ and $D_{opt}(C)$, which is what a practitioner actually wants: *given my budget, here is the model to train and how long to train it.*

One subtlety about $L$: the paper fits on smoothed **training** loss, not held-out test loss. That is legitimate here only because every run sees each token roughly once — with less than one pass over the corpus, training loss is an unbiased estimate of test loss. Remember this; it quietly becomes a limitation in Chapter 10.

### 1.2 The constraint: where $C \approx 6ND$ comes from

The constraint surface is what makes the problem tractable, so it is worth deriving rather than memorizing.

Consider one parameter of the network and one token flowing through it. That parameter is used in exactly one multiply-accumulate: one multiplication, one addition. That is **2 FLOPs**. Summing over all $N$ parameters, a forward pass costs about $2N$ FLOPs per token.

The backward pass computes two sets of gradients — with respect to the activations and with respect to the weights — so it costs roughly twice the forward pass, or $4N$ per token. Total: $6N$ FLOPs per token.

Multiply by $D$ tokens:

$$C \approx 6ND$$

**Worked example.** Chinchilla has $N = 7 \times 10^{10}$ parameters and was trained on $D = 1.4 \times 10^{12}$ tokens:

$$C = 6 \times (7\times10^{10}) \times (1.4\times10^{12}) = 5.88 \times 10^{23} \text{ FLOPs}$$

The paper quotes Gopher's training budget as $5.76 \times 10^{23}$ FLOPs. Those agree to about 2%, which is the point of the whole experiment: Chinchilla was built to consume the same compute as Gopher and nothing more.

Appendix F gives the fuller accounting — separate terms for embeddings, the Q/K/V projections, the $QK^\top$ logits, the softmax, the attention-weighted sum over values, the output projection, the feed-forward block, and the final logits. Every one of those terms is a component you met in Lecture 1. Table A4 reports that the full count lands within about 10% of $6ND$ for models from 73M to 6.8B parameters, so the shortcut is safe.

The consequence of the constraint is the thing to internalize: **at fixed $C$, $N$ and $D$ are inversely proportional.** Doubling the model halves the data. There is no free lunch anywhere on this surface.

**Check yourself.** (a) GPT-3 is 175B parameters trained on 300B tokens. What is its approximate training budget in FLOPs? (b) If you kept that budget fixed but halved the model to 87.5B, how many tokens could you afford?

------------------------------------------------------------------------

## 2. What the field believed in March 2022

### 2.1 Kaplan et al.'s answer

Lecture 3 covered Kaplan et al. (2020), which established that loss falls as a power law in model size, dataset size, and compute over many orders of magnitude. It also answered the allocation question — and its answer was lopsided.

Kaplan et al. found $N_{opt} \propto C^{0.73}$ and $D_{opt} \propto C^{0.27}$. In practical terms: a **10× increase in compute should buy a 5.5× larger model but only 1.8× more data.**

The reasoning the field drew from this was straightforward and, on its own terms, correct: if parameters convert compute into performance nearly three times as efficiently as data does, then when you get more compute, you should overwhelmingly spend it on parameters. A closely related conclusion from that paper — that large models should *not* be trained to convergence — reinforced the same behaviour, since stopping early is exactly what you do when data is the thing you are not buying.

### 2.2 What that belief produced

The result was a remarkable convention. Nearly every large model trained between 2020 and 2022 saw roughly 300 billion tokens, regardless of size.

| Model          | Parameters | Training tokens | Tokens per parameter |
|----------------|------------|-----------------|----------------------|
| LaMDA          | 137B       | 168B            | 1.2                  |
| GPT-3          | 175B       | 300B            | 1.7                  |
| Jurassic-1     | 178B       | 300B            | 1.7                  |
| Gopher         | 280B       | 300B            | 1.1                  |
| MT-NLG         | 530B       | 270B            | 0.5                  |
| **Chinchilla** | **70B**    | **1.4T**        | **20**               |

Look down the middle column. As models grew 4× from GPT-3 to MT-NLG, the token count did not move — it went slightly *down*. The data axis had effectively been frozen while the parameter axis absorbed every additional FLOP.

The final column is not in the paper, but it is the number to carry in your head, and it is why the last row looks like a different species.

**Check yourself.** Kaplan et al. say 10× compute → 5.5× parameters and 1.8× tokens. Verify that these two exponents are consistent with the constraint $C \approx 6ND$. (Hint: $5.5 \times 1.8 \approx 10$, and $0.73 + 0.27 = 1$ — this is not a coincidence, and you should be able to say why in one sentence.)

------------------------------------------------------------------------

## 3. The crux: learning rate schedules

This chapter is short, and it is the whole disagreement.

### 3.1 The mechanism

Both papers train language models and fit power laws to the results. They reach opposite conclusions. The difference is almost entirely an artifact of how training curves were sampled.

Kaplan et al. used a **single cosine learning rate schedule** for each model, set to a horizon of roughly 130B tokens, and then read intermediate points off the training curve to obtain (loss, FLOPs) pairs at smaller token budgets.

Here is why that is a problem. A cosine schedule anneals the learning rate from its maximum down to a small final value over the length of the cycle. Halfway through a 130B-token cycle, the model is still running at a substantial learning rate — it has not been annealed, and it is sitting at a noticeably higher loss than it would be if training had been *designed* to end there.

So the loss you read at the halfway point is **not** the loss of a model trained on 65B tokens. It is the loss of a model that is midway through a longer journey and has not yet been allowed to settle. It is an overestimate.

Now trace the consequence. Those overestimates occur specifically in the low-$`D`$ regime. Training on less data therefore looks worse than it actually is. If short runs look bad, the fit concludes that marginal compute should go to parameters rather than tokens — which is exactly the $a = 0.73$ result.

Hoffmann et al. instead **match the cosine cycle length to the intended token budget** for every run, so each data point is a model that was actually trained to finish where it finished.

### 3.2 The supporting evidence

This is not asserted; Appendix B tests it. They sweep cosine cycle lengths of 1×, 1.1×, 1.25×, 1.5×, 2×, and 5× the number of training steps. Overshooting the cycle length by more than about 25% produces clear degradation in final loss. They also note that decaying the learning rate by 10× beats decaying to zero slightly, and that decaying by only 5× is clearly worse.

A nice internal consistency check: in Approach 1, every point selected onto the compute-optimal frontier falls within the last 15% of its training run. The best models at any FLOP count are the ones that were just about to finish — precisely what the matched-schedule rule predicts.

### 3.3 The secondary difference: model scale

A smaller but real difference is the size range. Most of Kaplan et al.'s runs were under 100M parameters. In this paper, the majority of runs exceed 500M and reach 16B. That matters because Appendix E documents genuine **curvature** in the FLOP–loss frontier, so a straight line fitted to tiny models extrapolates badly.

**The practical lesson, worth stating plainly:** you must decide how long you are going to train *before* you start training, because the schedule is part of the experiment. A run whose schedule does not match its horizon is not a valid observation of that horizon.

**Check yourself.** (a) Suppose you correct Kaplan's bias only partially — you use matched schedules for large $D$ but keep intermediate readings for small $D$. Which direction does your estimate of $a$ move? (b) Why does this bias not simply cancel out across all the runs?

------------------------------------------------------------------------

## 4. Approach 1 — minimum over training curves

### 4.1 Procedure

Fix a family of model sizes from 70M to over 10B parameters. For each size, train **four** models with different token horizons, spanning a factor of 16×, each with a matched cosine schedule decaying 10×.

Smooth each training curve (Gaussian smoothing, window of 10 steps) and interpolate it. You now have, for every run, a continuous map from FLOPs spent to loss achieved.

Then take the **lower envelope**. At 1,500 logarithmically spaced FLOP values, ask: across all runs, which one achieves the lowest loss at this FLOP count? Record its model size and token count.

Finally, fit power laws $N_{opt} \propto C^{a}$ and $D_{opt} \propto C^{b}$ to those envelope points.

### 4.2 The mental picture

Picture a scatter of loss-versus-FLOPs curves, one per run, each starting high and descending. Small models descend fast early but flatten out at a high floor. Large models start slower — they burn compute per token — but keep descending past where small models have stalled.

The **envelope** is the curve you would trace with a pencil held underneath all of them. At each FLOP count it touches exactly one run, and which run it touches shifts steadily toward larger models as the budget grows. The rate of that shift *is* the exponent $a$.

### 4.3 Result

$a = 0.50$, $b = 0.50$.

Equal scaling. Double the compute, and you should build a model $\sqrt{2}$ times larger and feed it $\sqrt{2}$ times more data.

### 4.4 The head-to-head test

Section D.4 contains the cleanest single experiment in the paper. At a budget of $10^{21}$ FLOPs, Kaplan's method recommends a 4.68B-parameter model; Approach 1 recommends 2.86B. They trained both (4.74B and 2.80B actual), holding batch size, maximum learning rate, and depth-to-width ratio fixed to avoid confounds.

The smaller model won.

This is a genuine prediction-and-test at small scale, and it is worth more than the fitted exponents on their own, because it rules out the possibility that the two methods merely describe the data differently without disagreeing about anything observable.

**Check yourself.** Why does the envelope need four runs per model size rather than one long run per model size, given that Approach 1 reads intermediate points off the curves anyway?

------------------------------------------------------------------------

## 5. Approach 2 — IsoFLOP profiles

### 5.1 Procedure

Approach 1 asks a somewhat indirect question. Approach 2 asks the direct one: **fix the budget, vary the model size, see which size wins.**

Pick nine FLOP budgets from $6\times10^{18}$ to $3\times10^{21}$. For each budget, train a range of model sizes; the token count for each is then forced by the constraint ($D = C / 6N$). Set each cosine cycle to match. Plot final loss against model size.

### 5.2 Why there is a valley

The resulting curves are U-shaped, and the shape is the intuition for the entire paper.

On the **left** (too small): the model has plenty of tokens but not enough capacity. It cannot represent what the data is telling it, so loss is bounded below by the approximation error of a small hypothesis space. Extra tokens are wasted on a model that cannot use them.

On the **right** (too large): the model has ample capacity but has been starved of tokens, because the constraint forced $D$ down as $N$ went up. It is a large network that has barely been trained. Loss is bounded below by optimization, not capacity.

Somewhere between is a **minimum**. That the curves show a clear, well-resolved valley — rather than a flat region or a monotone slope — is what makes the question well-posed at all.

To locate the minimum precisely, they fit a **parabola** to each IsoFLOP curve in log-parameter space. Fitting a parabola is a local quadratic approximation near the bottom of the valley; it is a way of reading off the minimum's location that is robust to noise in individual runs, since no single run has to be the true optimum.

Then, as before, fit power laws to how the valley bottom moves as the budget grows.

### 5.3 Result

$a = 0.49$, $b = 0.51$.

Again, essentially equal scaling — obtained from a completely different sampling of the same space.

### 5.4 Does the dataset matter?

Appendix C repeats the IsoFLOP analysis on two other corpora:

| Dataset            | $a$ (parameters) | $b$ (tokens) |
|--------------------|--------------------|----------------|
| MassiveText (main) | 0.49               | 0.51           |
| C4                 | 0.50               | 0.50           |
| GitHub code        | 0.53               | 0.47           |

The conclusion holds across natural text and source code. The paper attaches a condition to this: it holds as long as you do not train for more than one epoch — a condition worth remembering for Chapter 10.

**Check yourself.** The IsoFLOP curves get shallower near the minimum at larger budgets. What does that imply about how precisely you need to hit $N_{opt}$ in practice — and about how confident you should be in an exponent fitted to those minima?

------------------------------------------------------------------------

## 6. Approach 3 — a parametric loss surface

### 6.1 The functional form

The first two approaches fit the *frontier* directly. Approach 3 is more ambitious: model the entire loss surface, then derive the frontier analytically.

$$\hat{L}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

Three terms, and each has a specific meaning drawn from a classical risk decomposition. Take them one at a time.

**$E$ — the irreducible term.** No predictor, however large and however well trained, can predict natural language perfectly, because language is genuinely stochastic. $E$ is the loss of the ideal predictor on the true distribution: the *entropy of natural text*. It does not depend on $N$ or $D$ because it is not a property of your model at all.

**$A/N^{\alpha}$ — the function-approximation term.** A transformer with $N$ parameters spans a restricted space of functions. Even trained perfectly on infinite data, the best function in that space falls short of the ideal predictor. The gap shrinks as $N$ grows, as a power law. The paper motivates the power-law form by analogy to approximation-rate results for neural networks.

**$B/D^{\beta}$ — the optimization/stochastic term.** You never reach the best function in your space either. You see $D$ samples exactly once and take a finite number of gradient steps. The residual gap is the suboptimality of stochastic first-order optimization, which is known to be bounded below by a $1/\sqrt{D}$-style rate.

The structure is clean: **one term you cannot beat, one term bought with parameters, one term bought with data.** The allocation question becomes: which of the two purchasable terms gives you more loss reduction per FLOP?

### 6.2 Fitting

Fit $(A, B, E, \alpha, \beta)$ by minimizing the **Huber loss** between predicted and observed log-loss, using L-BFGS from a grid of initializations.

Two choices deserve explanation.

*Why Huber rather than squared error?* Huber loss is quadratic for small residuals and linear for large ones, so a badly-fit point cannot dominate the objective. The paper found this mattered: with $\delta = 10^{-3}$, low-compute runs ($C \le 10^{21}$) that the model fits poorly get automatically down-weighted as outliers. Larger $\delta$ made the fit chase the small-compute regime and predict held-out large runs worse.

*Why a grid of initializations?* The objective is non-convex, so L-BFGS finds a local minimum that depends on where it starts. Sweeping initializations and keeping the best fit reduces that dependence. They report the best initialization was interior to the sweep, not on its boundary — a basic sanity check that the grid was wide enough.

The fitted result:

$$L(N, D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}}$$

The $E = 1.69$ nats is a genuinely interesting number — it is this method's estimate of the entropy of natural text under their tokenizer, an asymptote no amount of scaling will cross.

Note also what the paper says about the exponents: both are below $\tfrac{1}{2}$, and lower than one would like. Larger exponents would mean scaling buys more per unit; improving them is left as a challenge to future architectures and optimizers.

### 6.3 Deriving the frontier (do this one by hand)

This is the most valuable derivation in the paper, and the paper states the result without showing the work. Minimize $\hat{L}$ subject to $6ND = C$.

Substituting $D = C/(6N)$ and setting the derivative with respect to $N$ to zero gives the balance condition

$$\alpha A D^{\beta} = \beta B N^{\alpha}$$

Substitute $D = (C/6)/N$ and collect powers of $N$:

$$N^{\alpha + \beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta} \quad\Longrightarrow\quad N_{opt} = G\left(\frac{C}{6}\right)^{a}, \qquad D_{opt} = G^{-1}\left(\frac{C}{6}\right)^{b}$$

with

$$G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}, \qquad a = \frac{\beta}{\alpha + \beta}, \qquad b = \frac{\alpha}{\alpha + \beta}$$

**Read the exponents carefully, because they are counterintuitive.** The exponent governing *parameters* is built from $\beta$, the *data* exponent — and vice versa. Here $\alpha = 0.34 > \beta = 0.28$, meaning parameters reduce their term faster than data reduces its term. Yet the optimal policy spends the *larger* share of the budget on data ($b > a$).

The reason: at the optimum the two reducible terms must stay in balance. The term that shrinks more slowly is the one that needs more resource thrown at it to keep pace. Data is the stiffer axis, so data gets the bigger exponent.

**Worked example — verify the reported numbers.** With $\alpha = 0.34$ and $\beta = 0.28$:

$$a = \frac{0.28}{0.62} = 0.452, \qquad b = \frac{0.34}{0.62} = 0.548$$

The paper reports $a = 0.46$, $b = 0.54$. Your 0.452 rounds to 0.45, not 0.46 — and falls just outside the paper's own stated interval of $(0.454, 0.455)$.

Do not skip past that. The gap is entirely explained by rounding $\alpha$ and $\beta$ to two decimals before dividing; the true fitted values were a little different. But it tells you something important: **this formula is extremely sensitive to its inputs.** Two-decimal coefficients are not enough to reproduce the paper's own recommendations. Chapter 12 returns to this, because it turned out to matter more than anyone noticed at the time.

### 6.4 Result

$a = 0.46$, $b = 0.54$ — mild disagreement with Approaches 1 and 2, tilted slightly further toward data. The paper attributes this to the Huber loss down-weighting low-compute points, combined with the observed curvature in the frontier, which pulls the predicted $N_{opt}$ down at large budgets.

**Check yourself.** (a) What would the optimal policy be if $\alpha = \beta$ exactly? (b) If a new architecture doubled $\alpha$ while leaving $\beta$ fixed, would you train bigger models or smaller ones at fixed compute?

------------------------------------------------------------------------

## 7. Three methods, one answer

### 7.1 The comparison

| Approach | $a$ ($N_{opt}\propto C^a$) | $b$ ($D_{opt}\propto C^b$) |
|----|----|----|
| 1\. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2\. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3\. Parametric loss fit | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| **Kaplan et al. (2020)** | **0.73** | **0.27** |

Intervals are 10th–90th percentiles from bootstrapping (80% resamples, 100 times).

Three procedures, differing in what they sample, what they fit, and how they weight runs, land within 0.04 of each other — and nowhere near 0.73. Methodological agreement across genuinely different estimators is the strongest evidence in the paper.

### 7.2 What it means in practice

Approach 1's projections, with the tokens-per-parameter ratio computed:

| Parameters | FLOPs                 | Tokens | Tokens/parameter |
|------------|-----------------------|--------|------------------|
| 400M       | $1.92\times10^{19}$ | 8.0B   | 20.0             |
| 1B         | $1.21\times10^{20}$ | 20.2B  | 20.2             |
| 10B        | $1.23\times10^{22}$ | 205.1B | 20.5             |
| 67B        | $5.76\times10^{23}$ | 1.5T   | 22.4             |
| 175B       | $3.85\times10^{24}$ | 3.7T   | 21.1             |
| 280B       | $9.90\times10^{24}$ | 5.9T   | 21.1             |
| 1T         | $1.27\times10^{26}$ | 21.2T  | 21.2             |

The last column is roughly constant at ~20. This is the origin of the **"20 tokens per parameter"** rule of thumb that dominated the next two years of LLM training. Note that the paper never writes that rule down — it falls out of the table, and it is only constant because $a \approx b \approx 0.5$.

### 7.3 The verdict on existing models

Compare against Table 1 in Chapter 2. GPT-3 ran at 1.7 tokens/parameter, Gopher at 1.1, MT-NLG at 0.5. The recommendation is 20.

**Worked example (not in the paper — derive it yourself).** GPT-3's budget is $C \approx 6 \times 1.75\times10^{11} \times 3\times10^{11} \approx 3.2\times10^{23}$ FLOPs. Using $N_{opt} \propto C^{0.5}$ anchored on the 67B / $5.76\times10^{23}$ row:

$$N_{opt} \approx 67\text{B} \times \sqrt{3.2/5.76} \approx 50\text{B}, \qquad D_{opt} = \frac{C}{6N} \approx 1.1\text{T tokens}$$

For the same money, OpenAI could have trained a ~50B model on ~1.1T tokens instead of a 175B model on 300B tokens — a model less than a third the size that would also be cheaper to serve.

The paper also notes that a 1T-parameter model only becomes optimal at around $10^{26}$ FLOPs, over 250× Gopher's budget. Nobody in 2022 was close.

And the harder implication: 3.7T tokens for a 175B model, 5.9T for 280B. Those quantities of high-quality text did not obviously exist in curated form. **The bottleneck moves from engineering to data.**

**Check yourself.** MT-NLG is 530B parameters. Using the table, roughly what compute budget would make 530B the right size, and how does that compare to what it actually received?

------------------------------------------------------------------------

## 8. Chinchilla: building the prediction

### 8.1 The design

Predictions are cheap. The paper spends its remaining compute testing one.

For Gopher's budget, the three approaches predict an optimum between 40B and 70B parameters. They chose the top of that range — 70B, trained on 1.4T tokens — citing dataset and computational efficiency considerations. Same FLOPs as Gopher, one quarter the size, roughly 4.7× the data.

|                     | Gopher             | Chinchilla         |
|---------------------|--------------------|--------------------|
| Parameters          | 280B               | 70B                |
| Training tokens     | 300B               | 1.4T               |
| Layers              | 80                 | 80                 |
| $d_{model}$      | 16,384             | 8,192              |
| Attention heads     | 128                | 64                 |
| Key/value size      | 128                | 128                |
| Max learning rate   | $4\times10^{-5}$ | $1\times10^{-4}$ |
| Batch size (tokens) | 3M → 6M            | 1.5M → 3M          |

Note the shape of the shrink: **same depth, half the width.** Feed-forward size is $4 \times d_{model}$ throughout, and note that heads × key size = 64 × 128 = 8,192 = $d_{model}$, exactly as in Lecture 1.

**Worked example — verify the parameter count from Lecture 1's architecture.** A standard block has $4d^2$ in attention (Q, K, V, O projections) and $8d^2$ in the feed-forward layer (two matrices of size $d \times 4d$), so $12d^2$ per layer:

$$12 \times 8192^2 \times 80 \approx 6.4\times10^{10} = 64\text{B}$$

That is 64B, not 70B. Adding one further $d \times d$ projection per layer — consistent with the relative positional encoding Chinchilla inherits from Gopher — gives $13d^2L \approx 69.8\text{B}$. The same correction takes Gopher from 258B to 279B. Both land on their stated sizes, which is a good sign that the reconstruction is right. The paper does not spell this out; you get it by doing the arithmetic.

### 8.2 The confounds (read this part critically)

Chinchilla is *not* a clean A/B test against Gopher. Four things changed besides size and data:

1.  **AdamW instead of Adam.** Appendix G shows AdamW-trained models beat Adam-trained ones at both 417M and 1.4B scale. Curiously, the AdamW model only overtakes the Adam model about 80% of the way through the cosine cycle — so an early readout would have shown the opposite.
2.  **Higher-precision weights in the optimizer state** (bfloat16 forward/backward, float32 copy in the sharded optimizer state).
3.  **A modified SentencePiece tokenizer** without NFKC normalization — 94.15% vocabulary overlap with Gopher's, reportedly better for mathematics and chemistry.
4.  **A different sampling mix** over the same MassiveText corpus, reweighted for the larger token count.

The paper is upfront about all four and quantifies the optimizer effect. But when you read "Chinchilla beats Gopher," some unquantified share of that margin belongs to items 1–4 rather than to the scaling argument. In a seminar, this is the fair critique to raise; the response is that Approaches 1–3 and the $10^{21}$ FLOP head-to-head (§4.4) do not depend on Chinchilla at all.

**Check yourself.** Design the experiment that would isolate the scaling effect cleanly. Why was it not run?

------------------------------------------------------------------------

## 9. Results

Chinchilla is evaluated against Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and MT-NLG (530B) — every one of them larger.

### 9.1 Language modelling

Lower bits-per-byte on **all** Pile subsets versus Gopher. On WikiText-103, perplexity 7.16 versus Gopher's 7.75. Against Jurassic-1, Chinchilla wins on all but two subsets (`dm_mathematics` and `ubuntu_irc`).

The paper immediately attaches a caveat, and you should too: Chinchilla saw 4× more data, so **train/test leakage may inflate these particular numbers.** This is why the paper de-emphasizes them in favour of the tasks below.

### 9.2 MMLU

|                                   | Accuracy (5-shot) |
|-----------------------------------|-------------------|
| Random                            | 25.0%             |
| Average human rater               | 34.5%             |
| GPT-3                             | 43.9%             |
| Gopher                            | 60.0%             |
| **Chinchilla**                    | **67.6%**         |
| Forecasters' June 2023 prediction | 63.4%             |
| Average human expert              | 89.8%             |

Two things stand out. First, +7.6 points over a model 4× its size. Second, Chinchilla in March 2022 **beat the average expert forecast for June 2023** — a year of expected progress, delivered by reallocating an existing budget rather than by spending more.

Per-task: better on 51 of 57, tied on 2, worse on 4 (`college_mathematics`, `econometrics`, `moral_scenarios`, `formal_logic`). It exceeds 90% on four individual subjects, which the paper reports no prior model had done.

### 9.3 The rest of the evaluation suite

| Benchmark                     | Chinchilla | Gopher | Other        |
|-------------------------------|------------|--------|--------------|
| BIG-bench (avg, 62 tasks)     | 65.1%      | 54.4%  | —            |
| LAMBADA (0-shot)              | 77.4%      | 74.5%  | MT-NLG 76.6% |
| RACE-m (few-shot)             | 86.8%      | 75.1%  | —            |
| RACE-h (few-shot)             | 82.3%      | 71.6%  | —            |
| HellaSwag (0-shot)            | 80.8%      | 79.2%  | MT-NLG 80.2% |
| Winogrande (0-shot)           | 74.9%      | 70.1%  | MT-NLG 73.0% |
| BoolQ (0-shot)                | 83.7%      | 79.3%  | GPT-3 60.5%  |
| Natural Questions (64-shot)   | 35.5%      | 28.2%  | —            |
| TriviaQA unfiltered (64-shot) | 72.3%      | 61.3%  | GPT-3 71.2%  |
| TruthfulQA (0-shot)           | 43.6%      | 29.5%  | —            |

On BIG-bench, Chinchilla wins on 58 of 62 tasks. On Natural Questions it sets a new closed-book state of the art. The TruthfulQA jump of +14.1 points at 0-shot is notable because the original TruthfulQA paper argued that *larger* models are *less* truthful; this result suggests better modelling of the pre-training data alone can improve truthfulness, at least on this benchmark.

The recurring pattern: a 70B model outperforming a 530B model on nearly everything.

### 9.4 Bias and toxicity

**Winogender coreference** (does the model resolve pronouns to the right occupation?):

|                                      | Chinchilla | Gopher |
|--------------------------------------|------------|--------|
| All                                  | 78.3%      | 71.4%  |
| Male pronouns                        | 71.2%      | 68.0%  |
| Female pronouns                      | 79.6%      | 71.3%  |
| Neutral pronouns                     | 84.2%      | 75.0%  |
| Female "gotcha" (anti-stereotypical) | 76.7%      | 66.7%  |

Chinchilla improves everywhere, and improves most on the anti-stereotypical female cases (+10.0). But the improvements are *uneven* — male pronouns gain only 3.2 points. The paper's own reading is the right one: a more compute-optimal model is not automatically a fairer one, and the gains it does deliver are distributed unequally.

**Toxicity** is essentially unchanged: mean PerspectiveAPI score 0.087 for Chinchilla versus 0.081 for Gopher across 25,000 unprompted samples. Lower language-modelling loss does not move unconditional toxicity in either direction. This replicates Gopher's own finding that toxicity is largely independent of model quality.

**Check yourself.** Chinchilla's language modelling wins are discounted for possible leakage, but its MMLU wins are not. What property of MMLU justifies that difference in treatment, and how strong is that justification?

------------------------------------------------------------------------

## 10. Limitations

### 10.1 The ones the paper states

- **Only two comparable large runs.** Chinchilla and Gopher, with nothing at intermediate scale. The extrapolation from ≤16B models to 70B is a leap taken once.
- **Power laws are assumed, not derived.** All three approaches presuppose the functional form.
- **Observed curvature.** Appendix E fits separate lines to the first, middle, and last third of the frontier points and gets different slopes. If the frontier is concave in log space, the paper may *still* be overestimating optimal model size — the correction it proposes may not go far enough.
- **Single-epoch regime only.** Every analysis run sees each token about once. What happens on repeated data is unaddressed.

### 10.2 Two the paper underweights

- **Inference cost is entirely absent from the objective.** The optimization minimizes loss at fixed *training* compute. The paper notes in passing that a smaller model is cheaper to serve, but never puts serving cost in the objective function. Chapter 12 shows this omission is the thing that eventually superseded the rule.
- **The single-epoch assumption versus Chinchilla itself.** Appendix C's cross-dataset result is conditioned on staying under one epoch, and §5 says the analysis runs did. But Table A1 shows Chinchilla's own training repeated MassiveWeb 1.24× and Wikipedia 3.40×. The flagship model sits slightly outside the regime in which the scaling analysis was validated.

------------------------------------------------------------------------

## 11. Reading notes: internal wrinkles

Worth collecting, both because careful reading is the skill and because a few of these turned out to matter.

1.  **Model and token ranges disagree with themselves.** The abstract says 70M–16B parameters on 5–500B tokens; §1 says "under 70M" to over 16B on 5B–400B tokens. Table A9's smallest entry is 44M.
2.  **MMLU: 67.5% or 67.6%?** The abstract says 67.5%; Table 6 and §4.2.2 say 67.6%.
3.  **Natural Questions numbers conflict.** §4.2.6 quotes Gopher at 21% (5-shot) and 28% (64-shot); Table 9 gives 24.5% and 28.2%.
4.  **"Outperforms on all tasks" is not quite true.** §4.2.5 says Chinchilla outperforms Gopher on all common-sense tasks; Table 8 shows a tie on PIQA (81.8% both), and the table caption correctly says "matches or outperforms."
5.  **Projections in text ≠ projections in table.** §3.4 says a 175B model wants $4.41\times10^{24}$ FLOPs and 4.2T tokens, and a 280B model $\sim 10^{25}$ FLOPs and 6.8T tokens. Table 3 gives $3.85\times10^{24}$/3.7T and $9.90\times10^{24}$/5.9T.
6.  **GPT-3 changes size.** §5 calls it 170B; Table 1 says 175B.
7.  **Two broken equation references.** In §3.3 the efficient-frontier paragraph says the optimum balances the two terms in "Equation (3)" — Equation (3) is the Huber objective; the terms in question are in Equation (2). In §D.2, "the second term only depends on $D$" should read *third* term.
8.  **A missing minus sign.** Equation (6) defines the expected risk as $\mathbb{E}[\log f(x)_y]$ and calls it a cross-entropy to be minimized. Cross-entropy is $-\mathbb{E}[\log f(x)_y]$; as written, minimizing it would minimize likelihood.
9.  **Gopher's budget is quoted two ways.** Figures use $5.76\times10^{23}$ FLOPs; Appendix F notes their own more careful calculation gives $6.3\times10^{23}$. "Same compute as Gopher" is true to roughly ±10%, not exactly.
10. **A confidence interval that excludes its own point estimate.** In Table 2, Approach 1 reports $b = 0.50$ with a bootstrap interval of $(0.501, 0.512)$ — the reported value sits just below the interval's lower end.
11. **Approach 3's intervals are startlingly tight.** $(0.454, 0.455)$ for $a$, from roughly 400 training runs. Compare Approach 2's $(0.462, 0.534)$ on the same quantity. This is the loose thread that got pulled in 2024.

------------------------------------------------------------------------

## 12. Legacy

### 12.1 The immediate effect

This paper changed what the field built, quickly and almost completely. The "bigger is better" reflex was replaced by the 20-tokens-per-parameter rule, and the scarce resource was redefined from accelerators to **high-quality text**. The paper's own closing argument — that dataset collection and curation deserve the attention that architecture and systems engineering had been getting — reads as an accurate forecast of the next several years.

The second-order effect was commercial. A model a quarter the size is a quarter the inference cost, which makes deployment economics work at scales that a 280B model does not.

### 12.2 How the rule was then broken, deliberately

The Chinchilla objective minimizes loss per unit of *training* compute. It says nothing about the cost of serving a model to millions of users, and serving is where most lifetime compute goes for a deployed model.

So the successors intentionally over-trained. LLaMA's authors cited exactly this reasoning: the preferred model is not the fastest to train but the fastest at inference, and they observed a 7B model still improving past 1T tokens where Chinchilla would have recommended roughly 140B. Llama 2 trained on 2T tokens; Llama 3 8B on over 15T — on the order of 100× the Chinchilla-optimal ratio for its size.

Sardana et al.'s *Beyond Chinchilla-Optimal* (arXiv:2401.00448) formalized this by adding inference demand to the objective, and derived that a developer expecting substantial inference should train models **smaller and longer** than Chinchilla-optimal.

So the modern position is not that Chinchilla was wrong; it is that Chinchilla answered a narrower question than the one practitioners face. Chinchilla-optimal is training-optimal. Deployment wants lifetime-optimal, and that is a different optimum in the same direction — even further from where the field stood in 2021.

### 12.3 The 2024 replication

In April 2024, Besiroglu, Erdil, Barnett and You at Epoch AI published *Chinchilla Scaling: A replication attempt* (arXiv:2404.10102). Unable to obtain the original data, they reconstructed a subset from the paper's own figures and refit Approach 3.

They reported three problems: the published coefficients fit the reconstructed data poorly; the confidence intervals are implausibly narrow, of a width that would require hundreds of thousands of observations rather than roughly 400; and the scaling policy implied by the published parametric fit is inconsistent with Approaches 1 and 2 and with the 20-tokens-per-parameter ratio actually used to build Chinchilla — the fitted surface implies something closer to 70 tokens per parameter at Chinchilla's scale. One of the original authors subsequently acknowledged an optimizer-configuration error behind the over-tight intervals.

This is worth sitting with, because it is a better seminar discussion than either "landmark" or "debunked."

What did **not** survive: the specific fitted coefficients of Approach 3, and the confidence intervals in row 3 of Table 2. Anyone who took $(A, B, E, \alpha, \beta)$ off the shelf and built on it — several papers did — inherited a bad fit.

What **did** survive: Approaches 1 and 2, the near-equal scaling conclusion, the learning-rate-schedule critique of Kaplan et al., the $10^{21}$ FLOP head-to-head, and Chinchilla itself, which outperformed Gopher regardless of how the loss surface was parameterized. The headline claim rested on three legs, and the leg that broke was the one the headline did not depend on.

The methodological lesson generalizes past this paper: **an empirical result that is only supported by one estimation procedure is fragile, and one supported by three that disagree slightly is not.** The slight disagreement between $a = 0.50$, $0.49$, and $0.46$ was, in hindsight, informative — the outlier was the method that failed to replicate.

------------------------------------------------------------------------

## Glossary

**Compute-optimal** — For a fixed training FLOP budget, the $(N, D)$ pair minimizing final loss. Distinct from *best possible* (unbounded compute) and from *inference-optimal* (accounting for serving cost).

**IsoFLOP curve** — Final loss plotted against model size, holding total training FLOPs constant, with token count adjusted to compensate. U-shaped; its minimum is the compute-optimal model size for that budget.

**Efficient/compute-optimal frontier** — The locus of $(N, D)$ pairs that are optimal for some budget. A straight line in log-log space under a power-law model.

**Cosine cycle length** — The horizon over which a cosine learning-rate schedule anneals. Must match the intended token budget; overshooting by more than ~25% measurably degrades final loss.

**Envelope (of training curves)** — The lower boundary of a family of loss-vs-FLOPs curves; at each FLOP count it gives the best achievable loss over all runs.

**Bits per byte (bpb)** — Language modelling loss normalized by raw bytes rather than tokens, enabling comparison across models with different tokenizers.

**Huber loss** — A loss that is quadratic near zero and linear in the tails, so outliers cannot dominate a fit. Used here with $\delta = 10^{-3}$.

**Risk decomposition** — Splitting expected loss into irreducible (Bayes) risk, function-approximation error from limited capacity, and stochastic/optimization error from limited data and steps. The three terms of Equation (2).

**Entropy of natural text** — The $E$ term; the loss floor of an ideal predictor. Estimated here at 1.69 nats.

**Gotcha example** — A Winogender case where the correct coreference contradicts occupational gender stereotypes.

**Tokens-per-parameter ratio** — $D/N$. ~20 under Chinchilla-optimal training; 1–2 for pre-2022 models; 100–2000 for inference-optimized models like the later Llama series.

------------------------------------------------------------------------

## Review questions

**Foundational**

1.  Derive $C \approx 6ND$ from first principles. Where does the 6 come from, and what would it become if backward passes cost the same as forward passes?
2.  Kaplan et al. found $a = 0.73$; this paper finds $a \approx 0.50$. State the methodological difference in two sentences without using the phrase "learning rate."
3.  Explain why the IsoFLOP curve has a minimum rather than being monotone. What limits performance on each side of the valley?
4.  What does each of the three terms in $\hat{L}(N,D) = E + A/N^\alpha + B/D^\beta$ represent, and why does only one of them lack a dependence on $N$ or $D$?

**Analytical**

5.  Derive Equation (4) from Equation (2) and the constraint $6ND = C$. Then explain in plain language why the exponent for parameters is built from $\beta$ rather than $\alpha$.
6.  You have $10^{24}$ FLOPs. Using the table in §7.2, estimate $N_{opt}$ and $D_{opt}$. Now suppose you can only obtain half the tokens you need. What is the best thing to do with the surplus compute, and what does the loss surface predict you will lose?
7.  Why does the paper insist that its three approaches constitute stronger evidence than any one of them would alone? Construct a scenario in which all three would agree and all three would be wrong.
8.  The paper discounts its language modelling results because of possible train/test leakage, but not MMLU. Evaluate that decision.

**Critical**

9.  Chinchilla differs from Gopher in optimizer, precision, tokenizer, and data mix as well as in size and tokens. How much of the reported margin can you attribute to scaling, and which experiment in the paper is least vulnerable to this objection?
10. Appendix C validates the scaling result on datasets trained for under one epoch, but Chinchilla itself repeats Wikipedia 3.4×. Does this undermine the result, and what experiment would settle it?
11. In 2024 a replication found that Approach 3's published coefficients fit the data poorly and implied a token/parameter ratio of roughly 70 rather than 20. Does this invalidate the paper? Argue both positions, then state which parts of the paper each position leaves standing.
12. Llama 3 8B was trained at roughly 100× the Chinchilla-optimal token ratio. Is that evidence that Chinchilla was wrong, or evidence that it answered a different question? Write the objective function whose optimum Llama 3 is closer to.

------------------------------------------------------------------------

## Further reading

**The paper this one is arguing with**

- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*, arXiv:2001.08361 — Lecture 3. Re-read §5–6 with Chapter 3 above in hand; the schedule choice is visible in their methodology once you know to look.

**The model being displaced**

- Rae et al. (2021), *Scaling Language Models: Methods, Analysis & Insights from Training Gopher*, arXiv:2112.11446 — the source of Chinchilla's architecture, dataset (MassiveText), and entire evaluation suite. The "Lessons Learned" section explains the precision changes Chinchilla adopted.

**The critique**

- Besiroglu, Erdil, Barnett & You (2024), *Chinchilla Scaling: A replication attempt*, arXiv:2404.10102 — the Approach 3 replication. Short and readable; the reconstructed data is public.

**The successor objective**

- Sardana et al. (2024), *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448 — adds inference demand to the objective. The natural next step if Chapter 12.2 interested you.
- Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models*, arXiv:2302.13971 — the paper that chose to over-train on purpose, and said why.

**Adjacent**

- Clark et al. (2022), *Unified Scaling Laws for Routed Language Models*, arXiv:2202.01169 — the mixture-of-experts analogue, cited throughout §2 and §3.
- Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*, arXiv:2305.16264 — what happens when you run out of tokens and must repeat data, i.e. the multi-epoch regime this paper explicitly defers.
