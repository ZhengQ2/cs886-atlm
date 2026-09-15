# Attention Is All You Need

### A Course Textbook

Based on Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin (Google, 2017)

---

## Preface

This textbook accompanies a lecture on the Transformer architecture. It's meant to be read either alongside the lecture or independently as self-study. Each chapter builds on the last: Chapter 1 covers what came before attention, Chapter 2 covers attention's first appearance as an add-on to recurrent models, and Chapters 3–4 cover the Transformer itself and its consequences.

**Prerequisites:** basic neural networks, matrix multiplication, softmax, gradient descent.

**How to use this book:** work through it in order the first time. After that, the glossary and summary tables in Chapter 5 should be enough for review.

---

## Chapter 1 — The World Before Attention

To understand why the Transformer mattered, you first need to understand what it replaced. Before 2017, sequence modeling (translation, language modeling, speech recognition) was dominated by two families of architecture: **recurrent networks** and **convolutional networks**. Both process sequences fundamentally differently from attention, and both have real limitations that motivated everything that follows.

### 1.1 Recurrent Neural Networks (RNNs)

**The core idea:** process a sequence one element at a time, carrying a "memory" (hidden state) forward from each step to the next.

$$
h_t = f(h_{t-1}, x_t)
$$

At each timestep *t*, the model takes the current input `x_t` and the previous hidden state `h_{t-1}`, and produces a new hidden state `h_t`, meant to summarize everything relevant from the sequence so far.

**Mental picture:** reading a sentence left to right, updating a running summary in your head after each word. By the end of the sentence, that summary is supposed to contain everything you need.

**Plain RNNs vs. LSTMs/GRUs:** plain RNNs suffer badly from vanishing/exploding gradients over long sequences — the signal from early words gets washed out by the time you reach the end. LSTMs and GRUs add *gating mechanisms* — learned switches controlling what information gets kept, forgotten, or written at each step — allowing them to retain information over much longer sequences than a plain RNN can. By the mid-2010s, LSTMs and GRUs were the standard for state-of-the-art sequence modeling.

**The fatal limitation:** computing `h_t` requires `h_{t-1}`, which requires `h_{t-2}`, and so on back to the start. This is a *hard sequential dependency* — you cannot compute step 50 before step 49. That means:

- No parallelism within a single training example — you're stuck processing sequentially even on hardware built for massive parallel computation.
- This becomes especially painful for long sequences, where memory constraints also limit how much you can batch across different examples.

### 1.2 Convolutional Sequence Models

**The core idea:** slide a small, fixed-size filter (kernel) across the sequence, and at each position, compute a local weighted combination of nearby elements.

$$
y_t = \sum_{i=-k/2}^{k/2} w_i \cdot x_{t+i}
$$

**Mental picture:** sliding a 3-word window across a sentence; at each position, mixing those 3 words together with learned weights to produce a new representation for the center word, then moving the window over by one and repeating.

**Key property — locality.** A single convolutional layer only lets a token "see" its immediate neighbors (however wide the kernel is). To let far-apart positions interact, you either need a very wide kernel, or you need to *stack layers* so the effective receptive field grows with depth.

**Why this was attractive:** unlike recurrence, every position's convolution can be computed simultaneously — `y_t` doesn't wait on `y_{t-1}`, it only depends on nearby *inputs*. This gave convolutional sequence models (like ConvS2S and ByteNet) full parallelism during training, something RNNs couldn't offer.

**But there was still a cost:** because each layer only connects nearby positions, relating two *far apart* positions requires the signal to pass through multiple layers. The number of operations needed to relate two arbitrary positions grows with the distance between them — linearly for ConvS2S, logarithmically for ByteNet (which uses dilated convolutions that skip increasingly large gaps).

### 1.3 The Core Limitations — Summary

| | Sequential? | Parallelizable? | Ops to relate two distant positions |
|---|---|---|---|
| Recurrence (RNN/LSTM) | Yes — strictly step-by-step | No | Grows with sequence length |
| Convolution | No | Yes | Grows with distance (linear or log) |

Neither architecture could relate two arbitrary positions in a sequence with a *constant* amount of computation, regardless of how far apart they were. That gap is exactly what self-attention was built to close — and it's the thread that runs through the rest of this book.

**Check your understanding:**

- Why can't RNN computation be parallelized within a single example?
- Why does a convolutional layer need to be stacked to relate distant tokens?

---

## Chapter 2 — Attention Enters the Picture

Attention wasn't invented by the Transformer paper — it was already a widely used technique, always paired with an RNN. This chapter covers where it came from before Chapter 3 shows what happens when you strip the RNN away entirely.

### 2.1 The Bottleneck Problem in Seq2Seq

The standard architecture for tasks like translation (Sutskever et al., 2014) worked like this:

- An **encoder RNN** (usually an LSTM) reads the input sequence one token at a time and compresses the *entire* sentence into a single fixed-length vector — its final hidden state.
- A **decoder RNN** generates the output sequence, conditioned only on that one vector.

**The problem:** cramming an entire sentence — arbitrarily long — into one fixed-size vector is a severe bottleneck. Performance degraded badly on long sentences, because information from early in the sentence got diluted by the time it reached the final hidden state.

### 2.2 Bahdanau Attention (2014)

Bahdanau, Cho, and Bengio's "Neural Machine Translation by Jointly Learning to Align and Translate" introduced attention specifically to solve this bottleneck.

- The encoder is a **bidirectional RNN**, producing a hidden state for *every* input position — not just a final summary. The full sequence of encoder states is kept, not discarded.
- At each decoder step, instead of relying on one static context vector, the model computes a **fresh context vector** as a weighted sum over *all* encoder hidden states.
- The weights come from an **alignment model** — a small feed-forward network that scores how well the decoder's current state matches each encoder state, normalized with softmax. This is what later gets called *additive attention*: the compatibility score is computed by a feed-forward net with a hidden layer, not a dot product.
- That context vector feeds into the decoder RNN alongside its own previous hidden state, to help produce the next output token.

Recurrence is still doing the heavy lifting here — both encoder and decoder are RNNs. Attention is *bolted on* as a way for the decoder to "look back" at the right part of the input at each step, instead of relying on a single squashed vector.

### 2.3 Luong Attention (2015)

Luong, Pham, and Manning's "Effective Approaches to Attention-based Neural Machine Translation" simplified and extended this idea:

- **Global attention** — attends over all encoder states, like Bahdanau's, but simplifies the scoring function to a dot product or a bilinear ("general") form, instead of a full feed-forward alignment network. This dot-product form is what the Transformer paper later calls *multiplicative attention*.
- **Local attention** — attends only to a small window of encoder positions around a predicted alignment point, trading flexibility for lower compute cost.

### 2.4 What Is Attention, Really?

Strip away the specific formulas and here is the general idea:

> **Attention is a way to compute a weighted average of a set of vectors, where the weights are calculated on the fly based on relevance — rather than being fixed in advance.**

That's the entire concept. Everything else (queries, keys, values, softmax, scoring functions) is machinery for computing those weights differentiably, so the whole thing can be trained end-to-end with backpropagation.

**Three ingredients:**

1. **A query** — "what am I currently looking for?"
2. **A set of candidates**, each represented two ways: a **key** (used to judge relevance) and a **value** (the content actually pulled in if relevant).
3. **A scoring + weighting step** — compare the query against every key to get relevance scores, normalize with softmax so they sum to 1, then take the weighted sum of the values.

**Worked example:** *"The trophy didn't fit in the suitcase because it was too big."*

To resolve "it," a query built from "it" is compared against keys from every other word. "Trophy" scores high, "suitcase" scores lower, function words like "the" score near zero. Softmax turns these into a distribution (say, 70% trophy, 15% suitcase, small amounts elsewhere), and the output for "it" becomes a blend dominated by "trophy"'s value vector. None of this is hand-coded — the projections that produce queries and keys are learned weights, trained so this pattern of matching emerges from data.

**What varies across different attention mechanisms:**

- *The scoring function* — Bahdanau uses a feed-forward network (additive); Luong and the Transformer use a dot product (multiplicative).
- *Where queries/keys/values come from* — attending across two different sequences (encoder→decoder) versus attending within one sequence (self-attention, see Chapter 3).
- *Whether recurrence is present at all* — this is the Transformer's whole departure.

**What attention is *not*:**

- Not a literal database lookup — nothing persists outside a single forward pass.
- Not inherently aware of order — a raw attention operation over a set of vectors doesn't know word order without help (see §3.8).
- Not hard selection — even irrelevant candidates get *some* nonzero weight from softmax.

**Check your understanding:**

- What problem does attention solve that a single fixed-length context vector cannot?
- What's the difference between additive and multiplicative attention scoring?

---

## Chapter 3 — The Transformer Architecture

### 3.1 Design Philosophy

Recall from Chapter 1: neither recurrence nor convolution could relate two arbitrary sequence positions with a constant amount of computation. The Transformer's proposal:

> Drop recurrence and convolution entirely. Use **only attention** — computed in parallel — to model relationships between all positions in a sequence.

This is the origin of the paper's title: attention is no longer an add-on to a recurrent model (as in Chapter 2), it *is* the model.

### 3.2 Scaled Dot-Product Attention

This is the Transformer's specific attention mechanism — essentially Luong-style dot-product attention, with one crucial addition.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Step by step:

1. **`QKᵀ`** — dot product of every query with every key → a matrix of raw compatibility scores.
2. **`/√d_k`** — scale down. Without this, for large key dimension `d_k`, dot products grow large in magnitude and push softmax into a region with tiny gradients. This scaling is exactly what closes the performance gap that made unscaled dot-product attention underperform additive attention at high dimensions.
3. **`softmax(...)`** — normalize each row into a probability distribution summing to 1.
4. **`× V`** — take the weighted sum of value vectors. This is the output.

### 3.3 Multi-Head Attention

**The problem with a single attention function:** it produces one weighted average per token. But relationships between words are multi-faceted (syntax, coreference, topical similarity), and a single softmax distribution forces the model to average over all of it at once.

**The fix:** run several attention "heads" in parallel, each in its own learned subspace.

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Queries, keys, and values are each linearly projected into `h` smaller subspaces (different learned projections per head), attention runs independently in each, and the `h` outputs are concatenated and projected once more back to the model dimension.

**Paper's numbers:** `h = 8` heads, `d_model = 512`, so each head operates in `d_k = d_v = 512/8 = 64` dimensions. Total compute is comparable to one full-size attention head — this trades width for representational diversity, not extra cost.

**Analogy:** 8 readers going over the same sentence, each tracking a different kind of relationship — then their notes get combined.

### 3.4 Where Attention Is Used — Three Flavors

| Type | Queries from | Keys/Values from | Purpose |
|---|---|---|---|
| Encoder self-attention | encoder | same encoder layer | every input token attends to all input tokens |
| Decoder self-attention (masked) | decoder | same decoder layer, masked | every output token attends only to *earlier* output tokens |
| Encoder-decoder attention | decoder | encoder output | every output token attends to the entire input sequence |

Masking in decoder self-attention works by setting illegal (future) positions to `−∞` before the softmax, so they receive zero weight — this preserves the autoregressive property (predictions for position *i* can only depend on known outputs before *i*).

*Note for later:* in decoder-only models like GPT, there's no separate encoder — so encoder self-attention and encoder-decoder attention both disappear, leaving only masked self-attention.

### 3.5 Encoder and Decoder Stacks

**Encoder:** 6 identical layers. Each layer: multi-head self-attention → feed-forward network. Each sub-layer is wrapped in a residual connection + layer normalization:

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

All sub-layers output dimension `d_model = 512` — necessary for the residual addition to work.

**Decoder:** also 6 identical layers. Each layer: masked self-attention → encoder-decoder attention → feed-forward network, each wrapped the same way.

### 3.6 Position-wise Feed-Forward Networks

Applied identically and independently to every position:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

Two linear layers with a ReLU in between. Input/output dimension 512, inner dimension 2048 — it expands, applies a nonlinearity, then contracts. Think of attention as the "mixing" step (across tokens) and the FFN as the "processing" step (within a token, after mixing).

### 3.7 Embeddings and Weight Tying

Tokens are converted to `d_model`-dimensional vectors via learned embeddings. The same weight matrix is shared between the input embedding, the output embedding, and the final pre-softmax linear layer — reducing parameters and tying input/output representations together. Embeddings are scaled by `√d_model`.

### 3.8 Positional Encoding

**The gap:** self-attention treats input as an unordered *set* — nothing in the attention formula itself encodes "token 3 comes before token 5." Word order clearly matters, so an explicit position signal is needed.

**The solution:** sinusoidal positional encodings, added directly to input embeddings:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

`pos` = position in the sequence, `i` = dimension index. Each dimension is a sinusoid; wavelengths form a geometric progression from `2π` to `10000·2π`.

**Why sine/cosine instead of a learned embedding?** This form should let the model easily learn to attend by *relative* position, since `PE(pos+k)` can be expressed as a linear function of `PE(pos)`. It also lets the model handle sequence lengths longer than any seen during training, since the function is defined for any position — not just ones with a learned lookup entry.

**Check your understanding:**

- Walk through the four steps of scaled dot-product attention out loud, in order.
- Why does splitting into 8 heads not increase total compute relative to one full-width head?
- What exactly would break if positional encoding were removed?

---

## Chapter 4 — Results and Legacy

### 4.1 Experimental Results

- **28.4 BLEU** on WMT 2014 English→German — beat previous best results, including ensembles, by more than 2 BLEU.
- **41.8 BLEU** on WMT 2014 English→French — new single-model state of the art.
- Trained in **3.5 days on 8 GPUs** — a fraction of prior best models' training cost, largely thanks to parallelization enabled by dropping recurrence.
- Generalized successfully to English constituency parsing, showing the architecture wasn't a translation-only trick.

### 4.2 Why It Mattered

The paper's central bet — that attention alone, without recurrence or convolution, could match or exceed the best sequential models — paid off in two ways simultaneously: better translation quality *and* dramatically faster training, because self-attention's constant-time relation between any two positions (see the Chapter 1 comparison table) is both more expressive and more parallelizable than what came before.

### 4.3 The Modern Landscape

This architecture — largely unmodified at its core — became the backbone of essentially every major language model since:

- **BERT** (2018): encoder-only Transformer, trained with masked language modeling.
- **GPT family** (2018–present): decoder-only Transformer, trained autoregressively. The "T" in GPT is literally *Transformer*.
- **Modern LLMs** (including Claude): decoder-only Transformers at much larger scale, often with efficiency tweaks (e.g. rotary position embeddings instead of sinusoidal, grouped-query attention) — but the scaled dot-product self-attention core from this paper is still there, underneath it all.

**Check your understanding:**

- In a decoder-only model like GPT, which of the three attention types from §3.4 survives, and which disappear? Why?

---

## Chapter 5 — Review

### Glossary

| Term | Meaning |
|---|---|
| **Query (Q)** | A vector representing "what am I looking for," used to search against keys |
| **Key (K)** | A vector representing "what do I contain," matched against queries |
| **Value (V)** | The actual content pulled in when a key matches a query |
| **Self-attention** | Attention where Q, K, and V all come from the same sequence |
| **Encoder-decoder attention** | Attention where Q comes from the decoder and K/V come from the encoder |
| **Multi-head attention** | Running several attention functions in parallel, each in a smaller learned subspace, then combining results |
| **Scaled dot-product attention** | Attention where compatibility is a dot product, divided by `√d_k` before softmax |
| **Additive attention** | Attention where compatibility is scored by a small feed-forward network (Bahdanau-style) |
| **Positional encoding** | A signal added to embeddings so the model can distinguish token order |
| **Residual connection** | Adding a sub-layer's input to its output (`x + Sublayer(x)`), easing training of deep stacks |
| **Layer normalization** | Normalizing activations within a layer to stabilize training |
| **Autoregressive** | Generating output one token at a time, each conditioned on previously generated tokens |
| **Masking** | Blocking certain positions from being attended to (e.g. future tokens in decoder self-attention) |

### Exam-style discussion questions

1. Why does self-attention need the `1/√d_k` scaling factor? What specifically breaks without it?
2. Why use multiple attention heads instead of one larger one — what's the actual representational trade-off?
3. Self-attention connects any two positions in one step (constant path length), unlike RNNs or CNNs. What's the practical cost of this benefit (think about compute as sequence length grows)?
4. Why is the decoder's self-attention masked, but the encoder's is not?
5. Trace the journey of a single input token through the full encoder stack — name every operation it passes through, in order.
6. Compare Bahdanau attention, Luong attention, and Transformer self-attention: what changed at each step, and why?

### Further reading

- Sutskever, Vinyals, Le (2014) — "Sequence to Sequence Learning with Neural Networks" (the RNN encoder-decoder baseline)
- Bahdanau, Cho, Bengio (2014) — "Neural Machine Translation by Jointly Learning to Align and Translate"
- Luong, Pham, Manning (2015) — "Effective Approaches to Attention-based Neural Machine Translation"
- Vaswani et al. (2017) — "Attention Is All You Need" (this course's primary text)
- Devlin et al. (2018) — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Radford et al. (2018) — "Improving Language Understanding by Generative Pre-Training" (GPT-1)
