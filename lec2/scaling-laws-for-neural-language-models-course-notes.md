# Scaling Laws for Neural Language Models

**Paper:** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (Johns Hopkins / OpenAI, January 2020) — arXiv:2001.08361
**Format:** self-contained course-note chapter
**Prerequisites:** the Transformer (Lecture 1) — decoder-only stacks, attention, the meaning of $d_{\text{model}}$, $n_{\text{layer}}$, $n_{\text{heads}}$, $d_{\text{ff}}$. GPT-2 and WebText (Lecture 2). Cross-entropy loss, SGD/Adam, batch size, the idea of a training step. Comfort reading log-log plots.

---

## Chapter 0. The one-sentence version

The test loss of a Transformer language model is a **smooth power law** in three numbers — parameter count, dataset size, and training compute — and nothing else you'd normally fuss over (depth vs. width, number of heads, feed-forward ratio) matters much by comparison.

Two consequences follow, and they're the reason this paper is on the syllabus:

1. **You can predict the loss of a model you haven't trained.** Train a ladder of small models, fit a straight line on log-log paper, extrapolate. This turns "how good will a $100\times$ bigger model be?" from a research question into arithmetic.
2. **You can compute how to spend a fixed compute budget.** Given $C$ FLOPs, there is an optimal model size, batch size, and number of steps. The paper's answer — spend almost all of a compute increase on *model size*, and stop training well before convergence — was the direct justification for the GPT-3 scaling decision, and was substantially revised two years later. Both halves of that sentence are worth understanding.

---

## Chapter 1. The world before this paper

### 1.1 Scaling was folklore

By late 2019 everyone in language modelling knew "bigger is better." GPT-2 (Lecture 2) had just demonstrated it across four model sizes. But *knowing* bigger is better tells you almost nothing operationally. It doesn't tell you:

- **How much** better. Is $10\times$ the parameters worth 2% loss or 30%?
- **What to buy.** Given a fixed number of GPU-hours, should you train a small model for a long time, or a huge model briefly?
- **When to stop.** Conventional practice was: train to convergence. Nobody had asked whether convergence is even the right target.
- **How much data you need.** More parameters presumably need more data — but how much more? Linearly? Quadratically?

Each of these was answered by intuition and hardware constraints, not measurement.

### 1.2 What prior work had established

The paper positions itself against a small existing literature on empirical scaling:

| Prior work | What it found | How this paper differs |
|---|---|---|
| Banko & Brill (2001), Goodman (2001) | Early observations of power-law-ish gains with dataset size | Restricted to classical NLP, no model-size axis |
| Hestness et al. (2017), Hestness et al. (2019) | Deep learning scaling is predictable; power laws in model and data size | Found data needing to grow **super-linearly** with model size; this paper finds **sub-linearly**. The disagreement matters |
| Rosenfeld et al. (2019) | An ansatz for generalization error jointly in model and data size, close in spirit | Appeared essentially concurrently |
| EfficientNet (Tan & Le, 2019) | Scale depth and width together, by tuned exponents, for vision | This paper finds that for language, the shape exponents are close to irrelevant — only total scale matters |
| McCandlish et al. (2018) | The gradient noise scale predicts a *critical batch size* | This paper adopts that machinery wholesale; it is load-bearing in Chapters 6–8 |

So the novelty is not "performance follows a power law." It's **precision and range** (trends spanning six to eight orders of magnitude depending on the axis), plus a joint law that lets you optimize allocation rather than just observe a trend.

### 1.3 The framing to hold onto

The paper's own analogy, in its discussion section, is the ideal gas law: a small set of macroscopic variables related in a universal way, independent of nearly all the microscopic details. They have the thermodynamics; the statistical mechanics underneath is missing. That honesty is a feature — hold onto it, because Chapter 10 is where it bites.

---

## Chapter 2. The measurement apparatus

You cannot fit a clean law to a dirty measurement. A surprising fraction of this paper's value is in careful definitions, so we do those first.

### 2.1 The loss

Autoregressive cross-entropy, in **nats**, averaged over a 1024-token context. That's it — one number, no downstream benchmarks. This is a deliberate choice: benchmark accuracies are noisy, saturating, and discontinuous, while log-likelihood is smooth and unbounded below.

**Reading aid:** $\text{perplexity} = e^L$. So $L = 3.0$ nats $\leftrightarrow \text{perplexity} \approx 20$; $L = 2.37 \leftrightarrow \text{perplexity} \approx 10.7$; $L = 1.7 \leftrightarrow \text{perplexity} \approx 5.5$. When you see a loss drop from $3.0$ to $2.7$ and think "that's nothing," translate: $\text{perplexity}: 20 \rightarrow 15$.

### 2.2 The data

WebText2 — WebText from GPT-2 (Reddit outbound links with $\geq 3$ karma, through Dec 2017) extended with links from Jan–Oct 2018. Totals: 20.3M documents, 96 GB of text, $1.62 \times 10^{10}$ words, which the GPT-2 byte-level BPE tokenizer (vocab 50,257) turns into **$2.29 \times 10^{10}$ tokens** — of which $6.6 \times 10^8$ are held out as test. Additional evaluation sets: Books Corpus, Common Crawl, English Wikipedia, and a collection of internet books.

Remember 22B tokens. It comes back in Chapter 5 as the thing that limits their largest models.

### 2.3 The definition of "model size" — and why embeddings are excluded

This is the single most important methodological choice in the paper.

Define **$N = \text{non-embedding parameter count}$**:

> $$N \approx 2 d_{\text{model}} n_{\text{layer}} \left(2 d_{\text{attn}} + d_{\text{ff}}\right) = \mathbf{12 n_{\text{layer}} d_{\text{model}}^2}$$

where the second form assumes the standard shape $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$. Biases and layer-norm parameters are dropped as sub-leading.

**Worked example.** Take $(n_{\text{layer}}, d_{\text{model}}) = (12, 768)$. Then $12 \times 12 \times 768^2 = 12 \times 12 \times 589{,}824 = \mathbf{84.9\text{M}}$. That's the "85M" model that appears throughout their figures. Take GPT-2 XL's $(48, 1600)$: $12 \times 48 \times 2{,}560{,}000 = \mathbf{1.47\text{ billion}}$ — the familiar "1.5B."

Now the important part. The embedding matrix has $n_{\text{vocab}} d_{\text{model}} \approx 50{,}257 d_{\text{model}}$ parameters and positional embeddings add $n_{\text{ctx}} d_{\text{model}}$. These are **excluded from $N$**.

Why does this matter so much? Look at what happens either way:

| Parameter count used | What the data looks like |
|---|---|
| **With** embeddings | Loss appears to depend on *depth* as well as parameter count; models of different depth trace different curves |
| **Without** embeddings | Curves for depths from 2 to 200+ layers collapse onto a **single** trend |

The reason is mechanical. For a small model, the embedding matrix dominates the parameter count — a $(2, 128)$ model has $\sim 300\text{K}$ parameters, of which $\sim 6.4\text{M}$ would be embeddings — so "parameter count" for small models mostly measures vocabulary, not capacity. Strip the embeddings and the confound goes away.

This is a genuinely deep point disguised as bookkeeping: **the clean law only exists in the right variable.** Fit in the wrong variable and you conclude, wrongly, that architecture matters.

> ⚠️ This choice is also, per later work, one of the reasons this paper's compute-optimal conclusion disagrees with Chinchilla's. See Chapter 11.

### 2.4 The compute estimate: why $C \approx 6NBS$

Every FLOP count in the paper rests on this. Derive it once and you'll never have to memorize it:

1. In a forward pass, each non-embedding parameter is used in one multiply-accumulate per token. A multiply-accumulate is **2** floating-point operations. $\rightarrow$ **$2N$ FLOPs per token, forward.**
2. The backward pass costs roughly twice the forward pass (gradients w.r.t. both activations and weights). $\rightarrow$ **$4N$ FLOPs per token, backward.**
3. Total: **$6N$ FLOPs per training token.**
4. A training run processes $B$ tokens per step for $S$ steps. $\rightarrow$ **$C \approx 6NBS$.**

The term this drops is the attention-over-context cost, $2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$ per token, which is negligible when $d_{\text{model}} \gg n_{\text{ctx}}/12$. With $n_{\text{ctx}} = 1024$ that means $d_{\text{model}} \gg 85$, which holds for everything but their tiniest models. (It emphatically does *not* hold for long-context models today — flagged in the paper's own caveats.)

Unit: **$1\text{ PF-day} = 10^{15} \times 24 \times 3600 = 8.64 \times 10^{19}\text{ FLOPs}$.**

**Worked example.** The 85M model, batch $2^{19} = 524{,}288$ tokens, 250,000 steps:
$C = 6 \times 8.49 \times 10^7 \times 5.24 \times 10^5 \times 2.5 \times 10^5 \approx 6.7 \times 10^{19}\text{ FLOPs} \approx \mathbf{0.77\text{ PF-days}}$.

### 2.5 Training setup

Adam, $2.5 \times 10^5$ steps, batch of $512$ sequences $\times 1024$ tokens $= 2^{19}$ tokens. Models above 1B parameters used Adafactor for memory reasons. Learning rate: 3000-step linear warmup then cosine decay to zero. They report that results at convergence were largely insensitive to the schedule — a claim worth remembering, because it is precisely the claim later work disputes.

The sweep: models from 768 to 1.5B non-embedding parameters, datasets from 22M to 23B tokens, plus deliberate variation of depth, width, heads, feed-forward ratio, context length, and batch size.

---

## Chapter 3. Shape barely matters

Before you're allowed to compress a Transformer to a single number $N$, you have to show that the other numbers don't matter. Chapter 3 is that permission slip.

The experiment: hold $N$ fixed, vary one shape hyperparameter, retrain. To hold $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$ fixed while changing $n_{\text{layer}}$, you compensate by changing $d_{\text{model}}$, and so on.

**The finding:** loss varies by only a few percent across a wide range of shapes. Specifically:

- **Aspect ratio** ($d_{\text{model}} / n_{\text{layer}}$) can vary by a factor of $\sim 40$ with only slight impact. A **$(6, 4288)$** model lands within **3%** of the loss of the $(48, 1600)$ shape used for GPT-2 — same parameter count, eight times shallower and nearly three times wider.
- Feed-forward ratio ($d_{\text{ff}} / d_{\text{model}}$) and attention head dimension ($d_{\text{model}} / n_{\text{head}}$) similarly produce only a few percent of loss variation.
- The exchange rate they quote: **22% additional compute compensates for a 1% loss increase.** So a 3% loss penalty from a bad aspect ratio is worth roughly a 70% compute penalty — real, but trivial next to the orders of magnitude the rest of the paper deals in.
- Only models with fewer than 2 layers, or extreme depth-to-width ratios, fall off the trend.

The offered explanation (borrowed from ResNet literature, Veit et al. 2016) is that deep Transformers may behave like ensembles of shallower ones, which would make depth partially fungible. The paper is careful to present this as a possible reason, not a demonstrated one.

**Mental picture:** imagine a budget of 85M parameters as a fixed volume of clay. You can shape it tall and thin or short and wide; the paper's finding is that the sculpture's *quality* depends on the volume, not much on the silhouette.

**Check yourself:** if aspect ratio were *not* nearly irrelevant, which of the paper's later results would collapse? (Answer: all of them. $L(N)$ wouldn't be a function, since each $N$ would have many losses depending on shape, and "optimal model size for a compute budget" would be an under-specified question.)

---

## Chapter 4. The three basic power laws

### 4.1 What a power law says

$L = (X_c / X)^\alpha$ means: plotted with $\log L$ against $\log X$, you get a straight line with slope $-\alpha$. The operationally useful restatement is that **multiplying $X$ by a constant multiplies $L$ by a constant** — the same factor every time, forever.

The three headline fits (Section 1.2 of the paper):

| Law | Regime in which it applies | Exponent | Scale constant |
|---|---|---|---|
| $L(N) = (N_c/N)^{\alpha_N}$ | Limited parameters, trained to convergence on enough data | $\alpha_N \approx \mathbf{0.076}$ | $N_c \approx 8.8 \times 10^{13}$ params |
| $L(D) = (D_c/D)^{\alpha_D}$ | Large model, limited data, early stopping | $\alpha_D \approx \mathbf{0.095}$ | $D_c \approx 5.4 \times 10^{13}$ tokens |
| $L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}$ | Limited compute, ample data, optimal model size, small batch | $\alpha_C^{\min} \approx \mathbf{0.050}$ | $C_c^{\min} \approx 3.1 \times 10^8$ PF-days |

The ranges over which these hold: $\sim 8$ orders of magnitude in $C_{\min}$, $\sim 6$ in $N$, $\sim 2$ in $D$.

### 4.2 Reading the exponents

The exponents are small, which is easy to misread as "scaling doesn't help much." Convert them into something you can feel:

| You multiply... | ...the loss multiplies by | i.e. |
|---|---|---|
| $N$ by $2$ | $2^{-0.076} = \mathbf{0.95}$ | 5% lower loss per doubling |
| $N$ by $10$ | $10^{-0.076} = \mathbf{0.84}$ | 16% lower per order of magnitude |
| $D$ by $10$ | $10^{-0.095} = \mathbf{0.80}$ | 20% lower per order of magnitude |
| $C_{\min}$ by $10$ | $10^{-0.050} = \mathbf{0.89}$ | 11% lower per order of magnitude |

**Worked example — the cost of one nat.** Suppose you're at $L = 3.0$ nats and want $L = 2.0$. You need a loss ratio of $2/3$, so:

- By parameters alone: $(3/2)^{1/0.076} = e^{0.405/0.076} \approx \mathbf{200\times}$ **more parameters**.
- By data alone: $(3/2)^{1/0.095} \approx \mathbf{71\times}$ **more data**.
- By compute: $(3/2)^{1/0.050} \approx \mathbf{3{,}300\times}$ **more compute**.

That single line is the whole economics of the field. Loss falls forever and it falls *slowly*, so every increment of quality costs an order of magnitude more than the last. Whether that trade is worth making depends entirely on whether small loss decreases produce large capability increases — a question this paper explicitly flags and does not answer.

### 4.3 What the scale constants do and don't mean

$N_c$, $D_c$, $C_c$ have **no fundamental meaning**. They change if you change the tokenizer or vocabulary size, because that rescales the loss by an overall factor. Only the exponents carry information about the underlying scaling behaviour. Students routinely over-interpret $N_c \approx 8.8 \times 10^{13}$ as some magic parameter count; it isn't one, it's a fit intercept in disguise.

### 4.4 Two caveats the authors state themselves

- These laws **must** break eventually: they predict $L \rightarrow 0$ as $N \rightarrow \infty$, and natural language has non-zero entropy. The paper sees no deviation at the top of its range, which is a statement about their range, not about infinity.
- The $\alpha_C$ fit here uses a *fixed* batch size, which is not optimal. Chapter 6 fixes this, producing the cleaner $C_{\min}$ variant. When the paper wants to extrapolate, it uses $L(C_{\min})$, not $L(C)$.

### 4.5 Two side results worth knowing

**Transfer to other distributions.** Models trained only on WebText2 were evaluated on Books, Wikipedia, Common Crawl, and internet books. Loss on those distributions is *also* a power law in $N$ with nearly the same exponent, offset by a roughly constant amount. More strikingly: out-of-distribution loss depends only on in-distribution loss — not on how long you trained, not on whether you've converged, not on depth. So "how well does it transfer" reduces to "how good is it," which is a much simpler thing to optimize.

**LSTMs vs Transformers.** With the same data and context, LSTMs match Transformers on *early* tokens in the context but plateau after fewer than ~100 tokens, while Transformer loss keeps improving across the full 1024-token context. So the Transformer's advantage is specifically **long-context use**, not raw parameter efficiency. This is a much sharper statement than "Transformers are better," and it's the kind of claim you can only make by looking at loss per token position rather than averaged loss.

Relatedly (Appendix D.5): loss at position $T$ in the context is itself a power law in $T$, with larger exponents for larger models — bigger models extract more from less context. And models trained with a tiny context ($n_{\text{ctx}} = 8$) beat the 1024-context models on the very first few tokens, since they spend all their capacity there.

---

## Chapter 5. Putting N and D together: the overfitting law

### 5.1 The joint equation

The two separate laws $L(N)$ and $L(D)$ don't tell you what happens when *both* are limited. The paper proposes:

> $$\mathbf{L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}} \qquad \text{(1.5)}$$

This looks arbitrary. It isn't — it's the simplest form satisfying three stated requirements.

**Term by term:**

- **$D_c/D$** — the data-starvation term. Send $N \rightarrow \infty$ and only this survives, giving $L \rightarrow (D_c/D)^{\alpha_D} = L(D)$. ✓
- **$(N_c/N)^{\alpha_N/\alpha_D}$** — the capacity term. Send $D \rightarrow \infty$ and this survives alone: $[(N_c/N)^{\alpha_N/\alpha_D}]^{\alpha_D} = (N_c/N)^{\alpha_N} = L(N)$. ✓ **The strange exponent ratio exists exactly so that the outer $\alpha_D$ cancels.** That's the whole reason it's there.
- **Outer exponent $\alpha_D$** — makes the limits work, and gives the expression a series expansion in integer powers of $1/D$.

**The three design principles**, in the paper's order:

1. *Rescaling.* A change of tokenizer multiplies the loss by an overall factor; the functional form must absorb that into $N_c$ and $D_c$. (Hence §4.3: those constants are not fundamental.)
2. *Correct limits.* $L(N, \infty) = L(N)$ and $L(\infty, D) = L(D)$. Note the consequence: knowing the two single-variable laws fully determines every parameter of the joint law.
3. *Analyticity at $D = \infty$*, i.e. a $1/D$ expansion with integer powers. Motivated by the idea that overfitting tracks the dataset's variance or signal-to-noise ratio, which scales as $1/D$.

The authors are candid that principle 3 has much weaker support than 1 and 2, and that it's what forces the asymmetric treatment of $N$ and $D$. Symmetric alternatives exist but lack the clean $1/D$ expansion and need an extra parameter. Their final defence is empirical: it fits.

### 5.2 The fitted values, and a trap

Fitting all four parameters to the joint data (Table 2):

| $\alpha_N$ | $\alpha_D$ | $N_c$ | $D_c$ |
|---|---|---|---|
| $0.076$ | $\mathbf{0.103}$ | $6.4 \times 10^{13}$ | $1.8 \times 10^{13}$ |

Note $\alpha_D = \mathbf{0.103}$ here versus **0.095** in the headline $L(D)$ fit, and $N_c = 6.4 \times 10^{13}$ versus $8.8 \times 10^{13}$. This is not a typo — the paper says so explicitly: those numbers come from fitting the full $L(N, D)$ surface rather than the one-dimensional slices. But it means **you must use a consistent set**, and the famous exponent $0.74$ comes from this table, not the headline one:

$$\frac{\alpha_N}{\alpha_D} = \frac{0.076}{0.103} = \mathbf{0.738 \approx 0.74} \qquad \text{(whereas } 0.076/0.095 \text{ would give } 0.80\text{)}$$

Setup: all models regularized with 10% dropout and early stopping when test loss stops improving. The fit is excellent except for the smallest dataset ($\sim 2 \times 10^7$ tokens, where one epoch is only 40 parameter updates) — plausibly a different regime entirely.

### 5.3 How much data do you actually need?

Define the overfitting penalty relative to infinite data:

> $$\delta L(N, D) \equiv \frac{L(N, D)}{L(N, \infty)} - 1$$

Substituting Eq. 1.5 gives

> $$\delta L \approx \left[1 + \left(\frac{N}{N_c}\right)^{\alpha_N/\alpha_D}\left(\frac{D_c}{D}\right)\right]^{\alpha_D} - 1 \qquad \text{(4.3)}$$

and the payoff is that **$\delta L$ depends on $N$ and $D$ only through the single combination $N^{0.74}/D$.** Two wildly different $(N, D)$ pairs with the same ratio overfit by the same amount. This is what the paper means by "universality of overfitting," and it's exactly the kind of collapse-onto-one-curve result that indicates you've found the right variables.

**Worked example — deriving the data requirement.** Run-to-run variation from random seeds is about 0.02 in the loss, so demand $\delta L \leq 0.02$:

1. $(1 + x)^{0.103} = 1.02 \rightarrow x = 1.02^{1/0.103} - 1 = e^{0.0198/0.103} - 1 = 1.212 - 1 = 0.212$
2. $x = (N/N_c)^{0.738} (D_c/D)$, so $D = (D_c/0.212) (N/N_c)^{0.738}$
3. With $N_c = 6.4 \times 10^{13}$ and $D_c = 1.8 \times 10^{13}$: $(6.4 \times 10^{13})^{0.738} \approx 1.55 \times 10^{10}$, so $D \approx (8.49 \times 10^{13} / 1.55 \times 10^{10}) N^{0.738} \approx \mathbf{5{,}500 N^{0.74}}$

which is the paper's Eq. (4.4): **$D \gtrsim 5 \times 10^3 N^{0.74}$.**

**Now use it.** For $N = 10^9$ parameters: $D \gtrsim 5{,}500 \times (10^9)^{0.738} = 5{,}500 \times 4.4 \times 10^6 \approx \mathbf{2.4 \times 10^{10}\text{ tokens} \approx 24\text{B}}$. WebText2 has 22B. So the paper's own statement falls right out: models below a billion parameters train on WebText2 with minimal overfitting, and their largest models hit mild overfitting. The dataset was sized, whether by design or luck, right at the edge.

### 5.4 The headline consequence

Since $D \propto N^{0.74}$ is **sub-linear**, every $8\times$ increase in model size requires only about a $5\times$ increase in data ($8^{0.74} = 4.7$) to keep overfitting at bay.

That single inequality is the paper's most-quoted and most-misused result. Two things to keep straight:

- It is an **overfitting-avoidance** bound, not a compute-optimality prescription. It tells you the minimum data to avoid a penalty, not the best data given a budget. The paper says this explicitly.
- It contradicts Hestness et al. (2017), who found super-linear scaling. That disagreement was left unresolved here.

---

## Chapter 6. Batch size, and the invention of $C_{\min}$

### 6.1 The problem

Every model in the sweep trained at the same batch size, $2^{19}$ tokens. But the right batch size isn't a constant — small models near convergence want small batches, large models early in training want huge ones. Comparing compute across runs at a fixed batch size therefore compares runs that are all inefficient, in *different* ways. That smears the $C$ axis.

### 6.2 The critical batch size

The paper imports the framework of McCandlish et al. (2018). The empirical relation, for training to any fixed target loss $L$:

> $$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1 \qquad \text{(5.1)}$$

Here $S$ is parameter updates, $E = BS$ is examples processed, $S_{\min}$ is the fewest possible steps (achieved at infinite batch size) and $E_{\min}$ the fewest possible examples (achieved at infinitesimal batch size).

**What this curve says.** It's a hyperbola trading steps against samples. You cannot have both minimum steps and minimum data — pushing one toward its floor sends the other to infinity. Define

> $$\mathbf{B_{\text{crit}}(L) \equiv \frac{E_{\min}}{S_{\min}}} \qquad \text{(5.2)}$$

Training at exactly $B_{\text{crit}}$ puts you at the balanced point of the hyperbola: **$2S_{\min}$ steps and $2E_{\min}$ examples** — double the minimum of each, which is the best compromise available. Below $B_{\text{crit}}$ you waste wall-clock time; above it you waste compute.

**Mental picture:** digging a trench with a crew. One worker takes forever (minimum total labour, maximum time). A thousand workers trip over each other (minimum time, enormous wasted labour). $B_{\text{crit}}$ is the crew size where doubling the crew stops halving the time.

### 6.3 $B_{\text{crit}}$ depends on the loss, not the model

The striking empirical result: **$B_{\text{crit}}$ is independent of model size.** A 3M-parameter model and an 85M-parameter model at the same loss want the same batch size. It depends only on how good you currently are:

> $$\mathbf{B_{\text{crit}}(L) \approx \frac{B_{\ast}}{L^{1/\alpha_B}}}, \qquad B_{\ast} \approx 2 \times 10^8\text{ tokens}, \qquad \alpha_B \approx 0.21 \qquad \text{(5.3)}$$

Since $1/\alpha_B \approx 4.76$, $B_{\text{crit}} \propto L^{-4.76}$, and the batch size roughly **doubles for every 13% decrease in loss** (because $2^{-0.21} \approx 0.87$).

**Worked example.** At $L = 3.0$ nats: $B_{\text{crit}} = 2.1 \times 10^8 / 3^{4.76} = 2.1 \times 10^8 / 187 \approx \mathbf{1.1 \times 10^6\text{ tokens}}$ — a million tokens per batch, matching the paper's remark that the largest models want 1–2M tokens per batch near convergence.

Why parameterize it so that $B_{\text{crit}}$ diverges as $L \rightarrow 0$? Because the gradient noise scale is expected to diverge as you approach the minimum loss, and $B_{\text{crit}}$ tracks that noise scale. They don't know the true floor $L_{\min}$ — they just know it's above zero and far below anything they reached — so they use a form that blows up at zero.

### 6.4 The two corrected quantities

With $B_{\text{crit}}$ in hand, translate any actual run into the idealized regimes:

> $$\mathbf{S_{\min}(S) \equiv \frac{S}{1 + B_{\text{crit}}(L)/B}} \qquad \text{— steps you'd need at } B \gg B_{\text{crit}} \qquad \text{(5.4)}$$
> $$\mathbf{C_{\min}(C) \equiv \frac{C}{1 + B/B_{\text{crit}}(L)}} \qquad \text{— compute you'd need at } B \ll B_{\text{crit}} \qquad \text{(5.5)}$$

Sanity check the limits: if you already train far above $B_{\text{crit}}$ then $B_{\text{crit}}/B \rightarrow 0$ and $S_{\min} \rightarrow S$ (you're already step-minimal). If you train far below it, $B/B_{\text{crit}} \rightarrow 0$ and $C_{\min} \rightarrow C$. Halfway, at $B = B_{\text{crit}}$, each gets divided by 2 — the factor-of-2 from §6.2.

From here on, every "compute" in the paper means $C_{\min}$. This is a normalization, not new data, and it's what makes the later trends clean.

---

## Chapter 7. Learning curves: $L(N, S)$

### 7.1 The equation

In the infinite-data limit, after an initial transient, loss as a function of model size and training steps fits:

> $$\mathbf{L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}} \qquad \text{(1.6 / 5.6)}$$

Fitted: $\alpha_N = 0.077$, $\alpha_S = \mathbf{0.76}$, $N_c = 6.5 \times 10^{13}$, $S_c = 2.1 \times 10^3$.

**Term by term.** This is a sum, not a product, and the two terms mean different things:

- **$(N_c/N)^{\alpha_N}$** — the **capacity floor**. The loss you'd reach with unlimited training. Doesn't move no matter how long you train.
- **$(S_c/S_{\min})^{\alpha_S}$** — the **optimization deficit**. How far above your own floor you currently sit. Falls with training and goes to zero eventually.

So: $\textit{loss} = \textit{where you could eventually get to} + \textit{how far you still are from it}$. A model's training curve is a descent toward its own private floor, and the floor is set by $N$ alone.

**The exponents are the punchline.** $\alpha_S = 0.76$ is **ten times larger** than $\alpha_N = 0.077$. Training-step returns are steep; parameter returns are shallow. Every additional order of magnitude of steps buys $10^{-0.76} \approx 0.17\times$ on the optimization term, while an order of magnitude of parameters buys only $0.84\times$ on the capacity term. That asymmetry is the engine of Chapter 8's conclusion — it's cheap to close the gap to your floor, and expensive to lower the floor, so you should buy a low floor and only partly close the gap.

The practical use: extrapolate the early part of a training curve and you can estimate the loss you'd reach much later, without paying for it.

The paper offers a speculative reading of $\alpha_S$: since the fit is best late in training when the loss is roughly quadratic, the power law should encode the spectrum of the Hessian, and the exponent's universality across model sizes suggests the Hessian eigenvalue density is roughly size-independent. Presented as a suggestion, not a result.

### 7.2 A lower bound on the early-stopping step

If you're data-limited, when should you stop? The reasoning: finite-$D$ and infinite-$D$ learning curves track each other until overfitting kicks in, so the stopping point should correspond to the point where the achievable gap has been closed. Inverting the $L(N, S)$ relation:

> $$\mathbf{S_{\text{stop}}(N, D) \gtrsim \frac{S_c}{\left[L(N, D) - L(N, \infty)\right]^{1/\alpha_S}}} \qquad \text{(5.7)}$$

Read it as: the smaller the overfitting gap between your finite-data loss and the infinite-data loss, the longer you should train. It's a lower bound and underestimates, because in reality finite-$D$ loss decreases more slowly, so you actually need more steps than this.

---

## Chapter 8. Optimal compute allocation — the paper's payoff

You have $C_{\min}$ FLOPs. Spend them on a bigger model, a bigger batch, or more steps. What's the split?

### 8.1 The empirical answer

Scan across model sizes at fixed compute; for each budget, one model size wins (Figure 11–14). Fitting the winners:

| Quantity | Scaling with $C_{\min}$ | Meaning |
|---|---|---|
| Optimal model size $N$ | $\propto C_{\min}^{\mathbf{0.73}}$ | $\sim 5\times$ larger per $10\times$ compute |
| Batch size $B$ ($\propto B_{\text{crit}}$) | $\propto C_{\min}^{\mathbf{0.24}}$ | $\sim 1.7\times$ larger per $10\times$ compute |
| Serial steps $S_{\min}$ | $\propto C_{\min}^{\mathbf{0.03}}$ | essentially **constant** |
| Data processed $D = BS$ | $\propto C_{\min}^{\mathbf{0.27}}$ | $\sim 1.9\times$ more per $10\times$ compute |
| Loss $L$ | $\propto C_{\min}^{\mathbf{-0.050}}$ | 11% lower per $10\times$ compute |

The exponent on steps is so small the paper notes it may be consistent with zero. Read the table as a whole: **a $10\times$ compute increase should buy you a $\sim 5\times$ bigger model, a $\sim 2\times$ bigger batch, and essentially no additional serial training time.** Scale the model, parallelize the data, don't train longer.

With prefactors (Appendix Table 6), for $C_{\min}$ in PF-days:

- $N_{\text{opt}} \approx 1.3 \times 10^9 C_{\min}^{0.73}$ parameters
- $B_{\text{crit}} \approx 2.0 \times 10^6 C_{\min}^{0.24}$ tokens
- $S_{\min} \approx 5.4 \times 10^3 C_{\min}^{0.03}$ steps (a lower bound)
- $D_{\text{opt}} \approx 2 \times 10^{10} C_{\min}^{0.27}$ tokens

**Worked example — you have 10 PF-days.**

1. $N_{\text{opt}} = 1.3 \times 10^9 \times 10^{0.73} \approx 1.3 \times 10^9 \times 5.4 \approx \mathbf{7 \times 10^9\text{ params}}$
2. Predicted loss: $L = (3.1 \times 10^8/10)^{0.050} = e^{0.05 \times 17.25} \approx \mathbf{2.37\text{ nats}}$ ($\text{perplexity} \approx 10.7$)
3. Cross-check via $B_{\text{crit}}$: $2.1 \times 10^8 / 2.37^{4.76} \approx 3.4 \times 10^6$ tokens — agreeing with $2.0 \times 10^6 \times 10^{0.24} = 3.5 \times 10^6$ ✓
4. Steps: $S_{\min} \approx 5.8 \times 10^3$; training *at* $B_{\text{crit}}$ needs $2S_{\min} \approx \mathbf{12{,}000\text{ steps}}$
5. Tokens seen: $B \times 2S_{\min} \approx 3.4 \times 10^6 \times 1.2 \times 10^4 \approx \mathbf{4 \times 10^{10}\text{ tokens}}$

Ratio: about **$6$ tokens per parameter.** Hold that number; Chapter 11 is about how it changed.

### 8.2 The theoretical answer, and why it agrees

The allocation isn't just a fit — it's derivable. Substitute $S = C/(6NB(L))$ into $L(N, S)$ and minimize over $N$ at fixed $C$. The optimality condition (Appendix B) yields:

> $$\mathbf{\alpha_C^{\min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N}} \qquad \text{(6.4)}$$

Plug in $\alpha_S = 0.76$, $\alpha_B = 0.21$, $\alpha_N = 0.077$: $1/(1.32 + 4.76 + 12.99) = 1/19.07 \approx \mathbf{0.052}$, against an empirical fit of 0.050. Similarly $N(C_{\min}) \propto C_{\min}^{\alpha_C^{\min}/\alpha_N} \approx C_{\min}^{0.71}$, against the empirical 0.73.

That a law fitted on learning curves predicts, within a few percent, a law fitted on compute-optimal frontiers is the strongest internal evidence in the paper. It's the difference between a curve-fit and a framework.

Note the harmonic-sum structure: $1/\alpha_C^{\min}$ is the *sum* of the three reciprocal exponents, so the smallest exponent ($\alpha_N = 0.077$, contributing 12.99 of the 19.07) dominates. Compute-scaling is bottlenecked by the hardest axis, parameters — exactly as series resistances add.

### 8.3 Stop early — how early, precisely

Substituting the optimality condition back into the loss (Eq. B.5):

> $$L(N_{\text{eff}}(C), C) = \left(1 + \frac{\alpha_N}{\alpha_S}\right)L(N_{\text{eff}}, \infty)$$

Since $\alpha_N/\alpha_S = 0.077/0.76 \approx 0.10$: **compute-efficient training stops about 10% above the model's own converged loss.** Not 1%, not 0.1% — 10%, which by 2019 standards is nowhere near converged.

Compare to typical practice, taken as stopping 2% above convergence. Fixing the target loss, the two recipes relate as (Eqs. B.12–B.14):

| | Compute-efficient ($f = 10\%$) vs. conventional ($f = 2\%$) |
|---|---|
| Parameters | $(1.10/1.02)^{1/0.077} \approx \mathbf{2.7\times}$ **more** |
| Steps | $((1+1/0.10)/(1+1/0.02))^{1/0.76} \approx \mathbf{7.7\times}$ **fewer** |
| Total compute | $2.7 \times 0.13 \approx \mathbf{0.35 \rightarrow 65\%\text{ less}}$ |

Same loss, one third of the compute, by training a model $2.7\times$ larger for $7.7\times$ fewer updates. This is where "**convergence is inefficient**" and "**big models may be more important than big data**" come from.

### 8.4 How wrong can you be?

Reassuringly forgiving. Models between **$0.6\times$ and $2.2\times$** the optimal size reach the same loss with only $\sim 20\%$ extra compute. And a $2.2\times$ oversized model gets there in **45% fewer steps** for that 20% compute premium — which is a real option if you have the parallelism and want wall-clock speed. Undersizing, conversely, is attractive when you care about inference cost, which this paper does not model at all.

---

## Chapter 9. The contradiction, and a conjecture about language itself

This is the most interesting page in the paper, and the one most often skipped.

**Set up the collision.** Two of the paper's own results point in incompatible directions at very large scale:

1. **Overfitting control** (Chapter 5) demands $D \propto N^{0.74}$. Combined with $N \propto C_{\min}^{0.73}$, that's **$D \propto C_{\min}^{0.54}$**.
2. **Compute-efficient training** (Chapter 8) actually supplies data at only **$D \propto C_{\min}^{0.27}$** — and that's already the maximum possible rate, because it corresponds to a *single epoch* with no data reuse at all.

Data is needed faster than compute-optimal training can supply it. So compute-efficient training eventually runs into overfitting **even if you never repeat a single token.**

**Make it quantitative.** Once data-limited, loss should follow $L(D) \propto D^{-0.095}$, and with $D \propto C_{\min}^{0.27}$ that gives $L \propto C_{\min}^{-0.026}$ — shallower than the compute law's $C_{\min}^{-0.050}$. A shallower line and a steeper line must cross. The crossing point:

> $$\mathbf{C_* \sim 10^4\text{ PF-days}, \quad N_* \sim 10^{12}\text{ params}, \quad D_* \sim 10^{12}\text{ tokens}, \quad L_* \sim 1.7\text{ nats/token}} \qquad \text{(6.8)}$$

The paper stresses these values are highly uncertain — an order of magnitude either way, since they come from differencing two fitted exponents.

**Two readings, both offered:**

- *Conservative:* the scaling laws break down at or before this point. Something must give, and this estimates where.
- *Speculative:* the intersection is meaningful. If you can't push past $N_*$ without qualitatively different data requirements, perhaps at that point the model has extracted all the reliably-available information in natural language, making **$L_* \approx 1.7$ nats a rough estimate of the entropy per token of natural language**, and the loss curve would flatten there.

**A computation worth doing.** The paper notes WebText2 averages 4.3 characters per token. So $1.7\text{ nats/token} = 1.7/\ln 2 \approx 2.45\text{ bits/token} \approx \mathbf{0.57\text{ bits per character}}$. (That lands in the neighbourhood of classic information-theoretic estimates of written English — a comparison the paper doesn't make, but one that makes the conjecture feel less arbitrary.)

The closing move is elegant: they observe that adding constant noise to the data would shift all losses by an additive constant without moving the crossing point, so the *location* of the critical point may be meaningful even if the absolute $L_*$ is not.

---

## Chapter 10. What the authors say could be wrong

Appendix C is short, honest, and exactly what you want to discuss in a seminar. Their own list:

1. **No theory.** There is no theoretical account of any of these laws. The scaling with model size and compute is described as especially mysterious. Without a theory, it's hard to know when the laws can be trusted.
2. **$B_{\text{crit}}$ extrapolation is shaky** outside the loss range they explored — and $B_{\text{crit}}$ controls the whole time/compute tradeoff.
3. **The small-data regime is poorly characterized.** Their $L(N, D)$ fits fail for the smallest datasets, and they didn't explore regularization or data augmentation, which could change results qualitatively.
4. **$C \approx 6NBS$ ignores the context term.** Their compute accounting breaks when $n_{\text{ctx}} \gtrsim 12 d_{\text{model}}$ — which is precisely the long-context regime the field moved into.
5. **Hyperparameters may be under-tuned.** They tuned learning rate and tried schedules, but concede they may have neglected something important (initialization scale, momentum).
6. **Learning rate interacts with run length.** They note that near-convergence training may need a smaller LR to avoid divergence while short runs might tolerate larger ones — and that they **did not experiment with higher learning rates for runs that didn't go to convergence.** Sit with that one; it turns out to matter enormously (Chapter 11).

For reference, the LR heuristic they used: $\operatorname{LR}(N) \approx 0.003239 - 0.0001395 \log N$, which they note breaks down above $10^{10}$ parameters.

---

## Chapter 11. Legacy — and the correction

### 11.1 What it enabled

This paper turned scale from a gamble into a forecast. Its most visible consequence arrived four months later: GPT-3 (Brown et al., 2020) was a 175B-parameter model trained on roughly 300B tokens, and its design and loss predictions were argued for directly on the basis of these laws. Note the ratio — **under 2 tokens per parameter**, even more parameter-heavy than the 6 tokens/param in our Chapter 8 example, exactly as the $C^{0.73}$ vs $C^{0.27}$ split demands at larger budgets.

The framework generalized, too. Henighan et al. (2020) found similar power laws for autoregressive modelling of images, video, and math, supporting the paper's conjecture that this isn't specific to language.

And the discussion's aside — that smooth quantitative improvement can conceal qualitative change, "more is different" — became the seed of the whole emergent-abilities debate.

### 11.2 Where it was wrong: Chinchilla

In 2022, Hoffmann et al. ("Training Compute-Optimal Large Language Models," the Chinchilla paper) redid the compute-optimal analysis and reached a materially different conclusion: **$N$ and $D$ should scale roughly equally** with compute (exponents near 0.5 each, rather than 0.73 and 0.27), implying roughly **20 tokens per parameter** rather than $\sim 2$–$6$. Their demonstration was a 70B model trained on 1.4T tokens outperforming a 280B model trained on far less — same compute, quarter the size, better results.

| | Kaplan et al. (2020) | Chinchilla (2022) |
|---|---|---|
| $N \propto$ | $C^{0.73}$ | $\approx C^{0.5}$ |
| $D \propto$ | $C^{0.27}$ | $\approx C^{0.5}$ |
| Tokens per parameter (compute-optimal) | $\sim 2$–$6$ | $\sim 20$ |
| Practical upshot | Build huge models, train briefly | Build smaller models, train much longer |

**Why the disagreement?** Subsequent work attributing the gap points at methodology rather than at any deep disagreement about power laws — most prominently the learning-rate schedule. Recall §2.5: every run here used a schedule tuned for $2.5 \times 10^5$ steps. If you then *evaluate* a shorter run by reading a point off the middle of that curve, the short run is handicapped by a learning rate that hasn't decayed for the length of training it actually got. That systematically makes "train longer" look worse than it is, and so biases the optimum toward larger models. Two further contributors identified in follow-up analyses: the exclusion of embedding parameters from $N$, and insufficiently tuned warmup for small models. Kaplan et al.'s own caveat #6 above names the learning-rate issue as untested — they flagged the crack that later split.

**The right lesson is not "this paper was wrong."** The functional forms, the $L(N, D)$ collapse, the critical-batch-size framework, and above all the *methodology* — fit a ladder of small models, extrapolate, verify — all survived and are now standard practice. What was revised is one coefficient, arrived at with a confounded experimental protocol. That is the ordinary way empirical science works, and it's a better story for a seminar than either "landmark paper" or "debunked paper."

---

## Reading notes: internal wrinkles to expect

These are real inconsistencies in the text. Flagging them so you don't lose twenty minutes assuming you misread.

1. **$\alpha_D$ is $0.095$ in Eq. (1.2) and Table 5, but $0.103$ in Table 2.** Explained by the paper (different fits: 1-D slice vs. full surface), but you must use one set consistently. The famous $0.74$ exponent comes from the Table 2 pair.
2. **$N_c$ takes three values**: $8.8 \times 10^{13}$ (Table 5), $6.4 \times 10^{13}$ (Table 2, $L(N,D)$ fit), $6.5 \times 10^{13}$ (Table 3, $L(N,S)$ fit). Same reason.
3. **$\alpha_C^{\min}$ is quoted three ways**: $\approx 0.054$ in Eq. (6.4), $\approx 0.052$ in Eq. (B.7), $0.050$ as the fitted value in Table 5. Plugging Table 5's exponents into the formula gives $0.052$.
4. **$D_{\text{opt}}$ differs by a factor of 2** between Eq. (6.7) ($4 \times 10^{10} C^{0.26}$) and Table 6 ($2 \times 10^{10} C^{0.27}$). These are consistent if you track conventions: Eq. (6.7) computes data at $C = 2C_{\min}$ (training *at* the critical batch size), Table 6 quotes it at $C_{\min}$. Watch the factor of 2 whenever $C_{\min}$ appears.
5. **"Seven orders of magnitude"** (abstract) vs. **"six"** (§1.1 bullet) vs. the per-axis breakdown of 8/6/2 in §1.2. Not contradictory, just different quantities being counted.
6. **Appendix B.4 has unrendered placeholders** — it references "Figure X" and "Figure Y," and cites "A.1, A.6, A.9" where it means B.1, B.6, B.9. Leftovers from a renumbering. Nothing is missing mathematically.

---

## Glossary

| Term | Meaning |
|---|---|
| **Nat** | Unit of cross-entropy using natural log. $\text{Perplexity} = e^L$. $1\text{ nat} = 1/\ln 2 \approx 1.44\text{ bits}$ |
| **$N$ (non-embedding parameters)** | Parameter count excluding token and positional embeddings; $\approx 12 n_{\text{layer}} d_{\text{model}}^2$ for standard shapes. The variable in which the scaling law is clean |
| **$D$** | Dataset size in tokens |
| **$C \approx 6NBS$** | Estimated non-embedding training compute. $2N$ forward $+ 4N$ backward FLOPs per token, times tokens processed |
| **PF-day** | $8.64 \times 10^{19}$ FLOPs — one petaflop/s sustained for 24 hours |
| **Power law** | $L = (X_c/X)^\alpha$. Straight line on log-log axes; fixed multiplicative gain per fixed multiplicative cost |
| **$\alpha_N, \alpha_D, \alpha_C, \alpha_S, \alpha_B$** | Scaling exponents for parameters ($0.076$), data ($0.095$), compute ($0.050$ for $C_{\min}$), steps ($0.76$), batch size ($0.21$) |
| **$B_{\text{crit}}(L)$** | Critical batch size — the balanced point of the steps/samples tradeoff. Depends only on current loss, not model size |
| **Gradient noise scale** | Measure from McCandlish et al. (2018) that approximately predicts $B_{\text{crit}}$ |
| **$S_{\min}$** | Steps needed if trained at $B \gg B_{\text{crit}}$ (step-minimal regime) |
| **$C_{\min}$** | Compute needed if trained at $B \ll B_{\text{crit}}$ (compute-minimal regime). The x-axis the paper trusts for extrapolation |
| **Convergence factor $f$** | Fractional excess of current loss over that model's converged loss. Compute-efficient training sets $f \approx \alpha_N/\alpha_S \approx 10\%$ |
| **Compute-efficient frontier** | The set of $(N, B, S)$ choices minimizing loss at each compute budget |
| **Sample efficiency** | Loss reached per token processed. Increases with model size |
| **Universality of overfitting** | The observation that $\delta L$ depends on $N$ and $D$ only through $N^{0.74}/D$ |
| **WebText2** | Extended WebText: 20.3M docs, 96 GB, $2.29 \times 10^{10}$ BPE tokens |

---

## Review questions

**Mechanics**

1. A model has $n_{\text{layer}} = 24$ and $d_{\text{model}} = 1024$. Estimate $N$, then estimate the FLOPs to train it on 100B tokens, in PF-days.
2. Why is the factor in $C \approx 6NBS$ equal to 6 and not 2 or 3? Name each contribution.
3. Explain why including embedding parameters in $N$ makes depth *appear* to matter. Which model sizes are most distorted, and why?
4. Starting from $L(N, D) = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$, take the limits $N \rightarrow \infty$ and $D \rightarrow \infty$ and show you recover $L(D)$ and $L(N)$. Where exactly does the exponent ratio $\alpha_N/\alpha_D$ earn its place?

**Reasoning**

5. Your model sits at $L = 3.2$ nats and you want $2.8$. Using $\alpha_N = 0.076$, how many times more parameters (at fixed, ample data)? Using $\alpha_C^{\min} = 0.050$, how much more compute? Why are these so different?
6. You plan a $3 \times 10^{10}$-parameter model. Using $D \gtrsim 5 \times 10^3 N^{0.74}$, how many tokens do you need to avoid overfitting beyond seed noise? Is this the same as the amount of data you *should* train on? Explain the difference.
7. $B_{\text{crit}}$ is independent of model size but depends on the loss. Why is that the more useful parameterization in practice — and what does it imply about how batch size should change *during* a single training run?
8. In $L(N, S) = (N_c/N)^{\alpha_N} + (S_c/S_{\min})^{\alpha_S}$, $\alpha_S \approx 10\alpha_N$. Explain in one paragraph, with no algebra, why this ratio alone implies that compute-optimal training stops short of convergence.

**Synthesis**

9. Derive the $\sim 10\%$ figure: show that compute-efficient training ends at a loss $(1 + \alpha_N/\alpha_S)$ times the converged loss, and explain intuitively why the ratio of those two exponents sets the stopping point.
10. Reconstruct the Chapter 9 contradiction in your own words: which two scaling relations collide, why one is a hard ceiling rather than a choice, and what the crossing point is claimed to mean.
11. Suppose someone repeats this study but tunes the learning-rate decay separately for every run length. Which specific fitted exponent would you expect to move most, in which direction, and why? Connect your answer to the Kaplan/Chinchilla discrepancy.
12. The paper measures only cross-entropy loss, never downstream task performance, and explicitly flags that smooth loss improvements might hide qualitative capability jumps. Argue both sides: what does using loss as the sole metric buy you, and what does it cost?

---

## Further reading

**The machinery this paper stands on**

- McCandlish, Kaplan, Amodei, OpenAI Dota Team (2018), *An Empirical Model of Large-Batch Training* — the gradient noise scale and critical batch size. Chapters 6–8 here are unintelligible without it; read it if you want to understand where Eq. (5.1) comes from.
- Hestness et al. (2017), *Deep Learning Scaling is Predictable, Empirically* — the closest predecessor, and the one whose super-linear data scaling this paper contradicts.
- Rosenfeld, Rosenfeld, Belinkov, Shavit (2019), *A Constructive Prediction of the Generalization Error Across Scales* — a near-concurrent joint model-and-data ansatz.

**What it was applied to**

- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3) — the scaling laws cashed in. Read §6 of the Kaplan paper and then GPT-3's model-size table side by side.
- Henighan et al. (2020), *Scaling Laws for Autoregressive Generative Modeling* — the same exercise for images, video, and math, testing the universality conjecture from the discussion section.
- Hernandez, Kaplan, Henighan, McCandlish (2021), *Scaling Laws for Transfer* — extends the framework to fine-tuning and data transfer.

**The correction**

- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla) — the essential companion reading. Assign both papers together.
- Porian et al. (2024), *Resolving Discrepancies in Compute-Optimal Scaling of Language Models* — isolates the methodological causes of the Kaplan/Chinchilla gap (learning-rate decay not adapted to run length, embedding parameters, warmup). The best worked example of "the experiment, not the theory, was the problem."

**Theory attempts**

- Sharma & Kaplan (2020), *A Neural Scaling Law from the Dimension of the Data Manifold* — derives exponents from the intrinsic dimension of the data.
- Bahri, Dyer, Kaplan, Lee, Sharma (2021), *Explaining Neural Scaling Laws* — the most serious attempt at the "statistical mechanics" the discussion section asks for.

**The question the paper deliberately left open**

- Wei et al. (2022), *Emergent Abilities of Large Language Models*, and Schaeffer, Miranda, Koyejo (2023), *Are Emergent Abilities of Large Language Models a Mirage?* — the two sides of whether smooth loss improvement produces discontinuous capability. Directly downstream of the "more is different" paragraph.
- Sorscher et al. (2022), *Beyond Neural Scaling Laws* — evidence that better data selection can beat power-law scaling entirely, which is the most interesting way the framework could turn out to be too pessimistic.
