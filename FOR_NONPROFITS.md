# RAG-LITE-LR — A Guide for Non-Profit and Program Staff

This page explains what this tool does, when it's useful, and how to read its
results — without assuming any background in machine learning or software
engineering.

---

## What is this, in plain language?

RAG-LITE-LR is a **search assistant for a document collection you choose** —
for example, education research from [ERIC](https://eric.ed.gov), open-access
journal articles from [DOAJ](https://doaj.org), or your own grant reports and
program evaluations. You type a question in plain language — like "what
programs reduce chronic absenteeism in middle school?" — and it searches your
collection for the documents most relevant to that question.

It's different from a normal keyword search in a few ways:

- **It understands meaning, not just words.** If you ask about "programs that
  keep kids in school," it can find a document about "dropout prevention
  interventions" even though none of your words match exactly.
- **It "imagines" an ideal answer first.** Before searching, the system asks
  Claude (an AI assistant) to write a short hypothetical summary of what a
  perfect matching document might say, then searches for real documents that
  look similar to that summary. This is called "HyDE" and tends to surface
  more relevant results than searching with your raw question alone.
- **It combines two search methods.** One matches exact words and phrases
  (useful for program names, acronyms, specific terms); the other matches
  overall meaning and topic. Results from both are blended together.
- **It explains itself.** For each document returned, the system asks Claude
  to write a short note on *why* it might be relevant, plus two numeric
  scores (described below).

Think of it as an assistant that reads your question, searches your
collection two different ways, and hands you a short list with notes on why
each item might be worth reading.

---

## Choosing a research domain

If your organization's collection covers more than one program area (for
example, education research alongside environmental conservation research),
the "Research domain" dropdown in the sidebar lets you focus a search on
just one area — your question is matched only against documents tagged
with that domain, so results stay relevant to the topic you're working on.
Choose "All domains" to search everything at once.

## What you get back

For each search, you'll see a ranked list of documents (by default, 5). For
each one:

- **Title, authors, year, and source**
- **A link to the original document**
- **A relevance score** (1–10) — how directly this document addresses your
  question
- **A specificity score** (1–10) — how narrowly focused the document is on
  your exact question, versus a broad survey or tangential mention
- **A short written note** explaining the reasoning behind those scores

You'll also see how long the search took and how big your collection is.

---

## When to use this tool

- **Getting oriented on a program area or policy topic.** If staff are
  starting work on a new initiative, this can produce a useful starting
  reading list in a couple of minutes.
- **Finding evidence when you don't know the academic terminology.** Program
  staff often describe their work differently than researchers do — this tool
  bridges that gap.
- **Searching a focused collection you've built** — e.g., all evaluations of
  programs in a specific issue area, or a portfolio of grant reports.
- **Quick evidence checks** before writing a grant proposal, report, or
  funder update.

## When NOT to rely on this tool

- **As a substitute for a systematic evidence review.** This tool returns a
  *capped* shortlist based on similarity, not an exhaustive list of everything
  relevant.
- **For very recent events.** It only searches whatever collection you've
  built — if you haven't refreshed it recently, it won't know about new
  publications.
- **As the final word on "is this relevant."** The AI-generated scores and
  notes are a *starting point for your own judgment*. They can occasionally
  be overly generous, slightly off-topic, or based on a surface reading of
  the abstract.
- **For topics outside your chosen data source's coverage.** If your
  collection is built from ERIC (education-focused), don't expect strong
  results on, say, agricultural policy — see "Choosing a data source" below.

**Bottom line:** treat RAG-LITE-LR as a fast, AI-assisted starting point —
a way to generate candidates and gain orientation — not as the final or only
step in an evidence search.

---

## Choosing a data source

The quality of your results depends entirely on what's *in* your collection:

- **[ERIC](https://eric.ed.gov)** — free, no account needed, fast. Best for
  education, schools, early childhood, and related social-policy topics.
- **[DOAJ](https://doaj.org)** — free, no account needed, but slower (each
  search can take 30-45 seconds). Covers open-access journals across nearly
  every field — health, environment, social sciences, economics, and more.
- **Your own documents** — grant reports, internal evaluations, etc. can be
  converted into the same simple format (see the project README).

If your first searches return weak or off-topic results, the most likely
cause is that your collection doesn't yet contain documents on that topic —
try fetching more documents on that subject before assuming the search itself
is broken.

---

## What do the scores actually mean?

### Relevance score and specificity score (1–10, shown for each result)

These numbers are generated by an AI model (Claude) that reads your question
and the document's abstract, and rates:

- **Relevance (1–10):** How directly does this document address your
  question? A 10 means the document's central focus *is* your topic. Lower
  scores mean it's essentially unrelated. Middle scores (6-7) mean it touches
  on a related area but its main focus is something else.
- **Specificity (1–10):** How narrowly focused is the document on your exact
  question, versus being a broad survey or general overview?

**Important caveats:**
- These scores use a fixed scoring guide, so a "8" is intended to mean
  roughly the same thing across different searches — but they're still AI
  judgments, not measurements. Use them to compare results *within the same
  search*, not as precise universal numbers.
- Because the AI only sees documents that have already been shortlisted,
  scores tend to cluster on the higher end (most land in the 6–10 range). A
  "6" among your results may still be one of the *less* relevant items
  returned.
- Always use your own judgment, especially before citing a document in a
  report or proposal.

### Precision, Recall, NDCG, MRR (in the project's evaluation reports)

If you look at the technical evaluation results (`scripts/evaluate_retrieval.py`),
you may see metrics like "Precision@5," "Recall@5," "NDCG@5," and "MRR."
These describe how well the *system as a whole* performs on a small set of
test questions with known "correct" answers:

- **Precision@5** — Of the 5 results returned, what fraction were actually
  relevant (according to a hand-curated answer key)? Higher is better.
- **Recall@5** — Of all the known-relevant documents for a test question,
  what fraction showed up in the top 5? Higher is better.
- **NDCG@5** — Like recall, but also rewards putting the *most* relevant
  documents near the *top* of the list. Higher is better.
- **MRR (Mean Reciprocal Rank)** — On average, how close to the #1 spot did
  the *first* relevant document appear?

These numbers are measured against a small, hand-picked set of test
questions, each with a short list of documents known to be good answers.
Read them as a rough signal of quality for comparing setups — not as an
absolute guarantee of how well any single real-world search will perform.

---

## A note on cost and speed

Each search makes one or more calls to an AI service (Claude). With the
default settings (HyDE + relevance notes for 5 results), that's about 6 calls
per search. You can turn off HyDE and/or relevance notes in the sidebar (or
config) to reduce cost — see the README's "Cost controls" section. Heavy use
will incur API costs from your Anthropic account.

---

For more technical detail, see the [README](README.md).
