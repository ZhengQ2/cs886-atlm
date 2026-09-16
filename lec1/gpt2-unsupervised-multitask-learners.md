# Language Models are Unsupervised Multitask Learners (GPT-2)

**Paper:** Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI, 2019)
**Format:** self-contained course-note chapter
**Prerequisites:** the Transformer (Lecture 1) — self-attention, multi-head attention, residual + layer norm, decoder masking. Basic probability and the idea of maximum likelihood.

---

## Chapter 0. The one-sentence version

If you train a plain language model — next-token prediction, nothing else — on a large and *diverse* enough pile of text, it starts doing translation, summarization, and question answering **without ever being trained to do them**. You elicit the behaviour by writing the task into the prompt, not by changing the model.

Everything below is the argument for why that should be true, the engineering that made it testable, and the numbers showing how far it actually got in 2019 (answer: further than anyone expected on some tasks, and embarrassingly badly on others).

---

## Chapter 1. The world before this paper

### 1.1 Narrow experts

The dominant recipe in 2018 was: collect a labelled dataset for your task, train a model to imitate those labels, test on held-out examples drawn from the same distribution. It worked — and it produced systems the paper characterises as narrow experts rather than competent generalists.

The complaint is specifically about **brittleness**. These systems are sensitive to small shifts in the data distribution and to changes in how the task is specified. A reading comprehension model that scores well on its own test set can be broken by adversarially inserted distractor sentences; image classifiers fail on familiar objects in unfamiliar poses. Each of these is a symptom of the same thing: the model learned the dataset, not the task.

The authors' diagnosis: training on **single tasks over single domains** is a major contributor to this failure to generalize.

### 1.2 The obvious fix, and why it doesn't scale

The textbook answer to "your model only knows one task" is multitask learning — train on many (dataset, objective) pairs at once. In NLP circa 2019 this was barely off the ground: the two most ambitious efforts had reached 10 and 17 pairs respectively.

Here is the argument that kills it, and it's worth slowing down for because it's the intellectual hinge of the paper.

Take a meta-learning view. From that perspective, each (dataset, objective) pair is **one training example** — one sample from the distribution of possible tasks. Now ask: how many examples does a current ML system need to induce a function that generalizes? Hundreds to thousands.

So if you want multitask training to generalize *to new tasks* the way ordinary training generalizes to new examples, you would need hundreds to thousands of hand-built (dataset, objective) pairs. Every one of those requires humans to design an objective and label a corpus. That is not going to happen.

**Mental picture:** ordinary supervised learning needs thousands of labelled *sentences* to learn one task. Multitask learning, by the same logic, needs thousands of labelled *tasks* to learn task-generality. One of those is expensive; the other is impossible.

This is the gap. We want the generality that many tasks would buy, and we cannot afford to build the tasks.

### 1.3 The other trend line: transfer keeps getting more general

A second thread runs through the paper's introduction. Transfer learning in NLP had been steadily moving toward *less* task-specific machinery:

| Era | What gets transferred | What's still task-specific |
|---|---|---|
| word2vec / GloVe | Static word vectors | The entire architecture on top |
| CoVe / ELMo | Contextual representations from a recurrent net | The architecture on top |
| GPT-1 / BERT | A whole stack of self-attention blocks | Just a fine-tuning stage + output head |
| **This paper** | The whole model, used as-is | **Nothing** |

Read down the right-hand column. Each step deletes some hand-built, task-specific component. GPT-2's claim is that you can delete the last one — no parameter updates, no architecture modification, no fine-tuning. That's what **zero-shot** means here, and it's a stronger claim than the word is often used for today.

> **Check yourself:** Why is "no parameter or architecture modification" a much stronger claim than "we only needed 100 labelled examples"? What could you conclude from the first that you couldn't from the second?

---

## Chapter 2. The core idea: tasks are already in the text

### 2.1 Language modelling, formally

A language model estimates a distribution over sequences of symbols. Because language has a natural left-to-right order, we factor the joint probability into a product of conditionals:

$$p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \dots, s_{i-1})$$

Reading this term by term:

- $x$ is one document, a sequence of symbols $s_1 \dots s_n$ (for us, BPE tokens — see Chapter 4).
- Each factor asks: **given everything so far, what comes next?**
- The product runs over every position, so training signal comes from *every* token, not one label per document.

Why this factorization and not some other? Two practical payoffs. It makes sampling tractable — generate one token, append, repeat. And it gives you any conditional you want of the form $p(s_{n-k}, \dots, s_n \mid s_1, \dots, s_{n-k-1})$, i.e. "complete this prefix," for free. That second property is what makes the whole zero-shot programme possible, so hold onto it.

The expressiveness of the models computing these conditionals had just jumped, thanks to self-attention architectures like the Transformer. Lecture 1's architecture is a load-bearing prerequisite for this paper, not background colour.

### 2.2 From $p(\text{output} \mid \text{input})$ to $p(\text{output} \mid \text{input}, \text{task})$

Learning a single task is estimating $p(\text{output} \mid \text{input})$.

But a general system must handle many tasks, **even for the same input**. Give it the sentence "The movie was fine, I guess." — do you want a sentiment label, a French translation, or a paraphrase? The input alone doesn't say. So the system must model

$$p(\text{output} \mid \text{input}, \text{task}).$$

The interesting question is *how you condition on the task*. Prior work did it structurally:

- **Architecturally** — task-specific encoders and decoders, one per task.
- **Algorithmically** — an outer/inner loop, as in MAML.

The paper's move, following McCann et al.'s decaNLP, is to notice that **language already specifies tasks, inputs, and outputs, all as one sequence of symbols**. You don't need an architectural slot for "task." You write the task down.

A translation example becomes the sequence:

```
(translate to french, english text, french text)
```

A reading comprehension example becomes:

```
(answer the question, document, question, answer)
```

The "task" is now just more tokens, living in the same stream as everything else. There is no `task_id` input anywhere in the model.

### 2.3 The argument that unsupervised training subsumes supervised training

This is the theoretical heart, and it is short enough to state in full.

Suppose you format your supervised data as sequences, as above. The supervised objective is: predict the *answer* tokens correctly. The unsupervised objective is: predict *every* token correctly.

The supervised objective is therefore **the same objective, evaluated on a subset of the sequence**. So the global minimum of the unsupervised objective is also a global minimum of the supervised one. Perfect next-token prediction implies perfect task performance, because doing the task is a sub-case of predicting the text.

Notice what this argument does and does not buy you. It says the *optimum* is shared. It says nothing about whether you can get there. The paper is explicit that the problem becomes whether we can optimize the unsupervised objective to convergence in practice — and their preliminary experiments in this toy setup found that large models *do* learn multitask behaviour this way, but **much more slowly** than explicit supervision.

Keep that "much slower" in your pocket. It is the honest price of the whole approach, and it predicts almost everything in the results section: the tasks where GPT-2 does well are the ones where the web contains a lot of natural demonstrations; the tasks where it flops are the ones where it doesn't.

### 2.4 From a toy setup to "language in the wild"

The clean argument above assumes nicely formatted (task, input, output) triples. Real web text is not that. So the paper makes a **speculation**, and labels it as one:

> A language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences, in order to better predict them — regardless of how the text was obtained.

Why would it? Because demonstrations of tasks are *already scattered through ordinary text*, and modelling them is just part of modelling the text. The paper includes a table of examples found in its own training data — sentences containing English text next to its French equivalent, in the middle of news articles, product blurbs, and forum posts. Nobody labelled those as translation pairs. They're just there.

**Concrete mental picture.** Imagine a webpage that says: *"...as they say in French: 'Je ne suis pas un imbécile' [I'm not a fool]."* To predict the bracketed English, the model has to translate. There is no translation dataset, no parallel corpus, no supervision signal — but there *is* a gradient that rewards translating correctly. Multiply that by 40GB of text and you have an accidental, unlabelled, extremely noisy multitask dataset.

If the model does this, it is performing **unsupervised multitask learning**. That's the title, and now it should read as a literal description rather than a slogan.

An alternative was available: learn tasks interactively from dialogue. The authors considered it overly restrictive — the internet already contains vast information passively, with no interaction required.

> **Check yourself:** The paper's argument requires the training corpus to *contain* demonstrations of the task. Predict, before reading Chapter 6, which of these GPT-2 will do relatively well and which badly: (a) completing a sentence that needs long context, (b) translating English→French, (c) answering trivia questions. Then check your reasoning against the numbers.

---

## Chapter 3. WebText: buying diversity with a proxy for human judgement

The argument in Chapter 2 only works if the training data is *diverse*. A model trained only on news will only have seen the tasks people demonstrate in news. So the dataset is not incidental here — it's the experiment.

Most prior LM work trained on a single domain: news articles, Wikipedia, or fiction books. That is exactly the wrong shape for this hypothesis.

### 3.1 Why not just use Common Crawl

The obvious source of near-unlimited diverse text is a web scrape like Common Crawl. It is orders of magnitude larger than existing LM datasets. It is also full of garbage — prior work using it noted large quantities of documents that are largely unintelligible, and the authors observed the same in their own early experiments.

One prior team's workaround was to take only the subsample of Common Crawl most similar to their target dataset. That's pragmatic, but it's a form of cheating for this particular experiment: it means deciding in advance what tasks you care about. The whole point here is to avoid assumptions about which tasks will be performed.

### 3.2 The filtering trick

So they needed a *quality* filter that isn't a *topic* filter. The solution:

**Only scrape pages that a human already chose to share.**

Concretely: take all outbound links from Reddit that received at least 3 karma. The karma threshold is a heuristic for whether other users found the link interesting, educational, or just funny.

This is worth appreciating as a design pattern. Manual filtering of a full web scrape is prohibitively expensive, so they outsourced the curation to millions of people who had already done it, for free, for their own reasons. The paper's acknowledgements thank exactly those people.

It is also worth naming the obvious cost, which the paper does not dwell on: the resulting corpus inherits whatever Reddit's user base found worth sharing in 2017. "Diverse" here means diverse *relative to Wikipedia-only or news-only*, not representative of humanity.

### 3.3 The resulting corpus

| Property | Value |
|---|---|
| Source | Outbound Reddit links, ≥3 karma |
| Links collected | 45 million |
| After dedup + heuristic cleaning | slightly over 8 million documents |
| Size | 40 GB of text |
| Cutoff | No links created after Dec 2017 |
| **Wikipedia** | **Removed entirely** |

That last row matters for how you read the results. Wikipedia was dropped because it's a common data source for other NLP datasets, and including it would contaminate the evaluations — you couldn't tell zero-shot competence from having memorised the test set's source material. Removing it is a deliberate handicap that makes the results more believable.

---

## Chapter 4. Input representation: byte-level BPE

This chapter is a good example of a "boring engineering detail" that is actually load-bearing for the paper's central claim. Read it with that framing.

### 4.1 The requirement

The claim is "we can evaluate on any dataset without modification." That's only true if the model can assign a probability to **any string**. Standard LM preprocessing — lowercasing, tokenization, an out-of-vocabulary `<UNK>` token — restricts the set of strings you can model at all. If your vocabulary can't represent a benchmark's text, you can't evaluate on it honestly.

### 4.2 The two bad options

**Option A: bytes.** Model UTF-8 bytes directly. This perfectly satisfies the requirement: every string is representable, vocabulary size 256, no OOV ever. Problem: byte-level LMs at the time were not competitive with word-level LMs on large-scale benchmarks, and the authors reproduced that gap in their own attempts on WebText.

**Option B: word-level BPE.** Byte Pair Encoding interpolates between the two extremes — frequent sequences get merged into word-like units, rare ones stay character-like. But despite the name, reference BPE implementations usually operate on **Unicode code points**, not bytes. To cover all of Unicode you'd need a base vocabulary of over 130,000 symbols before adding a single merge — against the 32,000–64,000 total vocabularies typically used.

### 4.3 The fix, and the subtlety inside it

Apply BPE **at the byte level**. Base vocabulary: 256. Full Unicode coverage. Word-like units where the data warrants them.

But there's a wrinkle, and it's the kind of detail worth being able to reproduce on an exam. BPE builds its vocabulary with a **greedy frequency heuristic**, and applied naively to bytes it produces bad merges. The authors observed it learning many near-duplicate tokens for common words — separate merged tokens for `dog.`, `dog!`, `dog?` and so on, because each punctuation-suffixed form is itself frequent.

Why is that bad? Because vocabulary slots are a fixed budget. Spending four of them on trivial variants of "dog" means four fewer for genuinely distinct units, and it fragments the model's capacity across forms that should share representation.

**The fix:** prevent BPE from merging across **character categories** — don't let letters merge with punctuation. With one exception: spaces are allowed to merge, because that substantially improves compression while causing only minimal fragmentation of words across tokens.

*(This is why GPT-family tokenizers famously treat `" dog"` — with a leading space — as a distinct token from `"dog"`. It's a direct consequence of this exception.)*

**Worked trace.** Take the string `the dog!`:

1. Start as raw UTF-8 bytes — always possible, no OOV.
2. Merges apply within categories: `the`, `▁dog` (the space-prefixed exception in action).
3. `!` cannot merge into `▁dog`, so it stays separate.
4. Result: roughly `["the", "▁dog", "!"]` — word-level efficiency, byte-level generality, no wasted `▁dog!` slot.

The payoff, stated plainly: the model can assign a probability to any Unicode string, which means it can be evaluated on any dataset regardless of that dataset's preprocessing, tokenization, or vocabulary. Chapter 6's "state of the art on 7 of 8 datasets" depends on this being true.

---

## Chapter 5. The model: four sizes of one architecture

### 5.1 What changed from GPT-1

The architecture is a Transformer, largely following OpenAI's GPT-1, with a short list of modifications:

| Change | What it is | Stated reason |
|---|---|---|
| Layer norm moved to the **input** of each sub-block | Pre-norm instead of post-norm | Follows pre-activation residual networks |
| Extra layer norm after the final self-attention block | One more normalization at the top | — |
| Residual weights scaled by $1/\sqrt{N}$ at init ($N$ = number of residual layers) | Modified initialization | Accounts for accumulation on the residual path as depth grows |
| Vocabulary → 50,257 | Byte-level BPE (Ch. 4) | — |
| Context 512 → **1024** tokens | Longer window | — |
| Batch size → 512 | Larger batches | — |

The $1/\sqrt{N}$ init is worth one extra beat of intuition. In a residual network, each block *adds* to a running stream. Stack $N$ of them with no correction and the variance of that stream grows with $N$, so deeper models start out badly scaled before a single gradient step. Shrinking each residual branch by $1/\sqrt{N}$ keeps the accumulated signal roughly stable as you add depth. Note this is an explanation of the *mechanism*; the paper states the motivation as accounting for residual accumulation with depth and leaves it there.

### 5.2 The four models

Four LMs were trained at approximately log-uniformly spaced sizes:

| Parameters | Layers | $d_{\text{model}}$ | Identity |
|---|---|---|---|
| 117M | 12 | 768 | Equivalent to the original GPT |
| 345M | 24 | 1024 | Equivalent to BERT-large |
| 762M | 36 | 1280 | — |
| **1542M (1.5B)** | **48** | **1600** | **"GPT-2"** |

Two things to notice.

First, **"GPT-2" names only the largest model** in this paper. The other three are the controls. This is a scaling study wearing a model-release trench coat, and the smallest model being *exactly* GPT-1 is what makes the comparison clean.

Second, the learning rate for each model was tuned manually on a 5% held-out sample of WebText, and — the sentence everyone quotes — **all four models still underfit WebText**. Held-out perplexity was still improving with more training time when they stopped.

That's a remarkable admission to put in your own paper. It means every number in Chapter 6 is a *lower bound*, and it is the single most direct invitation to "just make it bigger" in the whole document.

> **Check yourself:** Why does including a model that exactly matches GPT-1's size make the scaling claim more credible than four arbitrary sizes would?

---

## Chapter 6. Results

Read this chapter with one question in mind: *does the web plausibly contain demonstrations of this task?* It predicts the results better than task difficulty does.

### 6.1 Zero-shot language modelling on 8 benchmarks

The first test isn't a downstream task at all — it's zero-shot *domain* transfer on the task the model was actually trained for. Train on WebText, evaluate perplexity on eight standard LM benchmarks, no fine-tuning.

This is only possible because of byte-level BPE (Ch. 4). But there's still a mismatch: these benchmarks contain aggressively standardized text — disconnected punctuation and contractions, shuffled sentences, even `<UNK>` tokens, which appear only 26 times in 40 billion bytes of WebText. So the authors apply **invertible de-tokenizers** that strip these artifacts. Invertibility matters: it means you can still compute a valid log-probability of the original dataset, so the comparison stays honest. This is a mild form of domain adaptation and it buys 2.5 to 5 perplexity.

| Benchmark | Metric | Prior SOTA | 117M | 1542M |
|---|---|---|---|---|
| LAMBADA | PPL ↓ | 99.8 | 35.13 | **8.63** |
| LAMBADA | ACC ↑ | 59.23 | 45.99 | **63.24** |
| CBT-CN | ACC ↑ | 85.7 | 87.65 | **93.30** |
| CBT-NE | ACC ↑ | 82.3 | 83.4 | **89.05** |
| WikiText-2 | PPL ↓ | 39.14 | 29.41 | **18.34** |
| Penn Treebank | PPL ↓ | 46.54 | 65.85 | **35.76** |
| enwik8 | BPB ↓ | 0.99 | 1.16 | **0.93** |
| text8 | BPC ↓ | 1.08 | 1.17 | **0.98** |
| WikiText-103 | PPL ↓ | 18.3 | 37.50 | **17.48** |
| 1BW | PPL ↓ | **21.8** | 75.20 | 42.16 |

**7 out of 8 datasets improved**, zero-shot. (Table 3 in the paper lists ten columns because LAMBADA and CBT contribute two metrics each.)

Three patterns to extract:

- **Small datasets gain most.** Penn Treebank and WikiText-2 have only 1–2 million training tokens. A model trained on 40GB of general text simply knows more English than anything you can fit from 2M tokens.
- **Long-dependency datasets gain most.** LAMBADA and the Children's Book Test were *built* to test long-range structure, and a 1024-token context plus scale is exactly the right medicine.
- **1BW is the one loss, and the failure is informative.** It's the largest dataset *and* has the most destructive preprocessing: sentence-level shuffling that removes all long-range structure. The model's biggest advantage is precisely the thing 1BW destroys.

Read the 117M column against the 1542M column across every row. That monotone improvement, on tasks nobody trained for, is the actual finding of the paper.

### 6.2 Where scale helps, and how much

Across tasks, increasing capacity improves zero-shot performance in a **log-linear** fashion. Performance on both the WebText train and test sets improves *together* as model size increases — which is both evidence against memorization (Ch. 7) and confirmation of underfitting.

### 6.3 The downstream tasks, one at a time

Each of these uses the same trick: get the model into a context where the desired behaviour is the natural continuation.

**Children's Book Test (cloze, 10 choices).** Score each candidate by computing the probability of that choice *plus the rest of the sentence* under the LM; pick the highest. New SOTA: **93.3% on common nouns, 89.1% on named entities**, closing most of the gap to human performance. Note the care taken: one CBT test book (*The Jungle Book*) was found in WebText, so they report validation-set results instead.

**LAMBADA (predict the final word, ≥50 tokens of context needed).** Perplexity **99.8 → 8.6**; accuracy **19% → 52.66%**. Then the interesting part: inspecting errors showed most predictions were *valid continuations of the sentence but not valid final words* — the model wasn't using the constraint that the word ends the sentence. Adding a stop-word filter as a crude stand-in for that constraint pushes accuracy to **63.24%**. This is a genuinely instructive failure: the model understood the content and missed the *format* of the task, which is a recurring theme in zero-shot evaluation.

**Winograd Schema Challenge (commonsense pronoun resolution).** Resolve the ambiguity by picking the reading the LM assigns higher probability. **70.70%**, +7% over SOTA. The paper immediately cautions that the dataset has only 273 examples and points readers to work on how to contextualize such results — a caveat worth imitating.

**CoQA (conversational reading comprehension).** Condition on the document, the conversation history, and a final token `A:`, then greedily decode. **55 F1** on dev — matching or exceeding 3 of 4 baseline systems that used 127,000+ manually collected QA pairs. The honest footnote: inspection suggests GPT-2 often leans on simple retrieval heuristics, such as answering a *who* question with any name from the document. Supervised SOTA (BERT-based) was near the human level of 89 F1.

**Summarization (CNN/Daily Mail).** Induce the behaviour by appending **`TL;DR:`** after the article, then sample 100 tokens with top-$k$ sampling at $k=2$, and take the first 3 sentences.

| System | R-AVG |
|---|---|
| Bottom-Up Sum (SOTA) | 32.75 |
| Lede-3 | 31.55 |
| Seq2Seq + Attn | 23.99 |
| **GPT-2 with `TL;DR:`** | **21.40** |
| Random-3 sentences | 20.98 |
| **GPT-2 without the hint** | **15.03** |

Be honest about this table when you teach it: GPT-2 barely beats picking three random sentences from the article. Qualitatively the outputs look like summaries, but they often fixate on recent content or confuse details — how many cars were in a crash, whether a logo was on a hat or a shirt.

The *interesting* number is the gap between the last two rows. Removing the `TL;DR:` hint costs **6.4 points**. Same model, same weights, same article — only the prompt changed. That's a clean demonstration that task-specific behaviour can be invoked in a language model **with natural language**, which is arguably a more important finding than the ROUGE score itself.

**Translation.** Seed the context with example pairs formatted `english sentence = french sentence`, then prompt with `english sentence =` and greedily decode.

| Direction | GPT-2 | Comparison |
|---|---|---|
| EN→FR | 5 BLEU | Slightly *worse* than word-by-word substitution with a bilingual lexicon |
| FR→EN | 11.5 BLEU | Beats several unsupervised MT baselines; far below 33.5 BLEU SOTA |

The asymmetry makes sense — FR→EN lets the model lean on its very strong English model for the output side.

What makes this result genuinely surprising is the data. Non-English webpages were **deliberately removed** from WebText. Running a byte-level language detector over the corpus found only **10MB of French** — roughly **500× smaller** than the monolingual French corpora used in unsupervised MT research. It learned that much translation from scraps that survived a filter designed to remove them.

**Question answering (Natural Questions).** Seed with example QA pairs to establish the short-answer format. Exact match: **4.1%**, versus 1.0% for a baseline that returns the most common answer per question type — so 5.3× a trivial baseline, and far below the 30–50% of retrieval-hybrid systems.

But look at the calibration. On the 1% of questions it is most confident about, accuracy is **63.1%**. The model knows when it knows. (The paper's footnote reports one author scoring 17 of 100 on random samples in the same setting — before correcting himself to 14.)

### 6.4 Summary of the picture

| Task | Verdict |
|---|---|
| Language modelling | SOTA on 7/8, zero-shot |
| Cloze / long-range (CBT, LAMBADA) | SOTA, large margins |
| Commonsense (Winograd) | SOTA, small dataset caveat |
| Reading comprehension (CoQA) | Competitive with 3/4 supervised baselines |
| Summarization | Barely above random sentence selection |
| Translation | Above trivial baselines, far below real MT |
| Open-domain QA | Weak in absolute terms, well calibrated |

The paper's own discussion is unusually restrained: while suggestive as a research result, the zero-shot performance of GPT-2 is still **far from usable** in practice. Any lecture on this paper that skips that sentence is misrepresenting it.

---

## Chapter 7. Generalization vs. memorization

Any time a model trained on a web scrape sets records on public benchmarks, the first question should be: *did it just memorize the test set?* The paper takes this seriously enough to devote a section to it, which in 2019 was not standard practice.

### 7.1 The method

1. Build **Bloom filters** containing 8-grams of WebText training tokens.
2. Normalize strings to lowercase alphanumeric words, single-space delimited (this favours recall — it catches near-matches, not just exact ones).
3. Set the false-positive rate to be upper-bounded by $10^{-8}$; verify empirically by generating 1M strings, of which zero were false hits.
4. For any dataset, measure: what fraction of its 8-grams appear in WebText train?

A Bloom filter is the right tool because it answers "have I seen this?" in constant space and time, with one-sided error — it can say yes when the answer is no, but never no when the answer is yes. For a contamination audit, that's the safe direction: you'd rather over-report overlap than miss it.

### 7.2 The finding, and the twist

| Test set of | Overlap with its *own* train set | Overlap with WebText train |
|---|---|---|
| PTB | 2.67% | 0.88% |
| WikiText-2 | 0.66% | 1.63% |
| enwik8 | 7.50% | 6.31% |
| text8 | 2.34% | 3.94% |
| WikiText-103 | 9.09% | 2.42% |
| 1BW | **13.19%** | 3.75% |

Common LM test sets overlap WebText train by 1–6%, averaging **3.2%**. The twist: many of these datasets overlap **their own training splits more** — averaging **5.9%**. The contamination that people worry about in web-scraped corpora was already present, and often worse, inside the curated benchmarks themselves. WikiText-103's 60-article test set contains an article that is also in its own training data.

Per-task audits:

- **Winograd:** 10 schemata had any 8-gram overlap; 2 spurious; of the remaining 8, only **1** appeared in a context that gave away the answer.
- **CoQA:** ~15% of news-domain documents are in WebText, worth ~3 F1 on those — about 0.5–1.0 F1 on the averaged metric. No training questions or answers leaked, since CoQA was released after WebText's link cutoff.
- **LAMBADA:** 1.2% average overlap. Excluding *every* example with any overlap moves perplexity 8.6 → 8.7 and accuracy 63.2% → 62.9%.

Conclusion: overlap gives a **small but consistent** benefit, not large enough to explain the results. The recommendation the paper leaves behind — n-gram overlap deduplication as a standard verification step when building new datasets — is arguably one of its more durable contributions.

### 7.3 The second line of evidence

Performance on WebText's own train and test sets is similar and improves *together* with model size. A memorizing model would pull away on train and stall on test. Instead both climb, which is the signature of underfitting rather than overfitting.

And on generated text: sampling from GPT-2 conditioned on held-out articles produced **less** verbatim overlap with the training set than the ground-truth continuations did. Over 30% of samples had no 8-gram overlap at all, while the median for real test-set text was 2.6%.

That said, memorization is not zero. Conditioned on the first sentence and a half of the Gettysburg Address — which appears roughly 40 times in WebText — an argmax decode reproduces the speech, typically drifting after 100–200 tokens. The pattern is what you'd expect: heavily duplicated text gets memorized; most text doesn't.

---

## Chapter 8. Why it mattered

### 8.1 What the paper itself claims

The conclusion is narrow and defensible: a large LM trained on a sufficiently large and diverse dataset performs well across many domains and datasets, and the diversity of tasks it can do zero-shot suggests high-capacity models trained to maximize likelihood on varied text **begin to learn** how to perform a surprising number of tasks without explicit supervision.

The discussion adds a useful reframing: these results may help explain **why pre-training works so well** generally. If, in the limit, a pre-training objective starts performing tasks *directly*, then fine-tuning is better understood as surfacing an ability that's partly already there than as teaching one from scratch.

Open questions the authors name themselves: it's unclear where the **ceiling with fine-tuning** is; unidirectional representations may still be a handicap relative to BERT; and many practical tasks remain no better than random.

### 8.2 What followed

Three threads run directly out of this paper, and they're worth naming because they're the reason it gets taught:

1. **Prompting as an interface.** `TL;DR:`, `A:`, and `english = french` are all prompt engineering, three years before the term existed. The 6.4-point summarization drop when the hint is removed is the first clean measurement of prompt sensitivity.
2. **Scaling as a research programme.** "Log-linear improvement with capacity" plus "still underfits" is an unusually explicit roadmap. GPT-3 (2020) followed it at 175B parameters and showed that few-shot in-context learning, which is weak here, becomes strong with scale. The Kaplan et al. scaling-law work made the log-linear observation precise.
3. **Contamination auditing.** The Bloom-filter overlap analysis became a standard section in large-model papers.

### 8.3 A note on the release

GPT-2 was also the paper that made staged release a norm — the full 1.5B model was not published at once, with the original release covering only the smallest model. Whatever you think of that decision, it is part of why this paper is cited outside ML, and it's reasonable to flag in a seminar. Just keep it separate from the technical claims; the paper's contribution stands or falls on Chapters 2–7.

---

## Glossary

| Term | Meaning |
|---|---|
| **Zero-shot** | Performing a task with **no parameter or architecture modification** — not "few labelled examples." Stronger than common usage. |
| **WebText** | 40GB / ~8M documents scraped from Reddit outbound links with ≥3 karma; Wikipedia removed. |
| **Byte-level BPE** | BPE over UTF-8 bytes (base vocab 256, final 50,257), blocked from merging across character categories except for spaces. |
| **Task conditioning** | Modelling $p(\text{output} \mid \text{input}, \text{task})$; here the task is specified in-sequence as text rather than architecturally. |
| **Perplexity (PPL)** | Exponentiated average negative log-probability per unit. Lower is better. |
| **BPB / BPC** | Bits per byte / bits per character — the same quantity in a different unit, used for enwik8/text8. |
| **De-tokenizer (invertible)** | A reversible transform removing preprocessing artifacts from benchmark text, so log-probabilities remain valid. |
| **LAMBADA** | Predict the final word of a passage needing ≥50 tokens of context. |
| **CBT** | Children's Book Test — 10-way cloze, split by word category (common nouns, named entities, verbs, prepositions). |
| **CoQA** | Conversational QA over documents from 7 domains; questions depend on dialogue history. |
| **Top-$k$ sampling** | Sample only from the $k$ highest-probability tokens. Used with $k=2$ for summaries, $k=40$ for the appendix samples. |
| **Bloom filter** | Probabilistic set-membership structure with one-sided error; used here for 8-gram contamination auditing. |

---

## Review questions

1. State the meta-learning argument for why hand-built multitask learning cannot scale. Why is it an argument about the *number of tasks* rather than the *number of examples*?
2. Explain why the global minimum of the unsupervised objective is also a global minimum of the supervised objective. What does this argument **not** establish, and where in the results does that gap show up?
3. Why did naive byte-level BPE produce separate tokens for `dog.`, `dog!`, and `dog?`, and why is that harmful? What rule fixes it, and what single exception is carved out?
4. WebText excludes Wikipedia. Give two distinct reasons this strengthens rather than weakens the paper's conclusions.
5. GPT-2 is the *only* model in the table that loses to prior SOTA on 1BW. Explain the loss using a property of the dataset, not a property of the model.
6. On LAMBADA, a stop-word filter raises accuracy from 52.66% to 63.24%. What does that tell you about the nature of the model's errors? Name another result in the paper with the same character.
7. Removing `TL;DR:` costs 6.4 ROUGE points with no change to weights or input article. What claim from Chapter 2 does this directly support?
8. GPT-2 reaches 4.1% exact match on Natural Questions but 63.1% on its most confident 1%. Why is the second number arguably the more interesting one?
9. Design a contamination audit for a new benchmark, borrowing the paper's method. Why a Bloom filter, and why normalize strings before hashing?
10. The authors report that all four models still underfit WebText. Identify three separate claims elsewhere in the paper whose interpretation changes once you know this.

---

## Further reading

- Vaswani et al. (2017) — *Attention Is All You Need* (Lecture 1; the architecture used here)
- Radford et al. (2018) — *Improving Language Understanding by Generative Pre-Training* (GPT-1; the 117M model is its twin)
- Devlin et al. (2018) — *BERT: Pre-training of Deep Bidirectional Transformers* (the bidirectional contrast the discussion section raises)
- McCann et al. (2018) — *The Natural Language Decathlon* (decaNLP; source of the "tasks as text" formatting)
- Sennrich et al. (2015) — *Neural Machine Translation of Rare Words with Subword Units* (original BPE)
- Hestness et al. (2017) — *Deep Learning Scaling is Predictable, Empirically* (the scaling analysis this paper extends into the 1B+ regime)
- Paperno et al. (2016) — *The LAMBADA Dataset*; Hill et al. (2015) — *The Goldilocks Principle* (CBT)
- Brown et al. (2020) — *Language Models are Few-Shot Learners* (GPT-3; the direct sequel)
