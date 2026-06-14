# Retrieval Evaluation

Results from running `scripts/evaluate_retrieval.py` against the 17 gold
queries in `tests/eval/gold_queries.yaml` (top-5, HyDE + Claude
justification enabled, `claude-haiku-4-5`) over the ~20,689-document
corpus. Full per-query detail (retrieved IDs, justifier scores) is written
to `outputs/eval_report.json` each time the script runs.

## Headline metrics (mean across 17 queries, top-5)

| Metric | Value |
|---|---|
| Precision@5 | 0.306 |
| Recall@5 | 0.239 |
| NDCG@5 | 0.363 |
| MRR | 0.600 |

Five of the 17 queries score 0 across the board (see "Weakest queries"
below); the remaining 12 queries average closer to P@5 ~0.43 and MRR ~0.85,
in line with prior results on the smaller corpus.

## Per-query results

| Query | P@5 | R@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| What programs help improve early literacy outcomes for low-income preschoolers? | 0.200 | 0.250 | 0.246 | 0.500 |
| How big a problem is summer learning loss for low-income students, and what can schools do about it? | 0.600 | 0.750 | 0.754 | 1.000 |
| What interventions actually reduce chronic absenteeism in K-12 schools? | 0.800 | 0.400 | 0.786 | 1.000 |
| What does the research say about school-based mental health programs and their effectiveness? | 0.200 | 0.250 | 0.390 | 1.000 |
| What does trauma-informed practice look like in schools and does it help students? | 1.000 | 0.556 | 1.000 | 1.000 |
| How can we help first-generation students get into and succeed in college? | 0.400 | 0.500 | 0.637 | 1.000 |
| What family and parent engagement strategies improve outcomes for students? | 0.400 | 0.250 | 0.553 | 1.000 |
| Do after-school programs improve academic outcomes for at-risk students? | 0.600 | 0.429 | 0.655 | 1.000 |
| What approaches help local communities participate effectively in conservation programs? | 0.000 | 0.000 | 0.000 | 0.000 |
| How are local communities adapting to climate change, and what strategies are working? | 0.000 | 0.000 | 0.000 | 0.000 |
| What does community-based participatory research look like in practice? | 0.000 | 0.000 | 0.000 | 0.000 |
| How can organizations evaluate the impact of their community health programs? | 0.200 | 0.091 | 0.131 | 0.200 |
| How can arts-based programs support youth development and community engagement? | 0.000 | 0.000 | 0.000 | 0.000 |
| What strategies help preserve and promote cultural heritage in local communities? | 0.000 | 0.000 | 0.000 | 0.000 |
| What does the research say about environmental health disparities in low-income or marginalized communities? | 0.200 | 0.111 | 0.339 | 1.000 |
| How do nonprofit organizations measure and improve their own effectiveness? | 0.400 | 0.333 | 0.345 | 0.500 |
| How can creative arts programs support young people's mental health and wellbeing? | 0.200 | 0.143 | 0.339 | 1.000 |

**Weakest queries:** five queries (conservation programs, climate adaptation,
community-based participatory research, arts-based youth development, and
cultural heritage) score 0 on every metric. In each case the retrieved
results are topically on-target (e.g. for "conservation programs", the
pipeline returns several community-based-conservation papers rated 8-10 on
relevance by the justifier) but don't overlap with the curated
`relevant_ids`. This looks like a gold-set-curation/corpus-density artifact
of the recently expanded corpus &mdash; these domains now contain large
clusters of similar papers, many near-equally relevant to a broad query
&mdash; rather than a HyDE or retrieval defect. Addressing it would likely
require either a re-ranker or further gold-set curation.

## Latency

| Metric | Value |
|---|---|
| Mean total | 5.64s |
| Min | 4.64s |
| Max | 7.82s |

Dense (~0.05s) and BM25 (~0.002s) retrieval are negligible; nearly all
latency comes from the Claude HyDE and justification calls.

## Justifier score calibration

| Score | n | Mean | Stdev | Min | Max |
|---|---|---|---|---|---|
| relevance_score | 85 | 8.93 | 1.14 | 4 | 10 |
| specificity_score | 85 | 7.38 | 1.36 | 5 | 10 |

Scores skew high and cluster tightly, which is expected for an
already-filtered top-5 set, but leaves limited headroom for the justifier
to discriminate between borderline results.

## Reproducing

```bash
python scripts/evaluate_retrieval.py
# or, to skip Claude justification calls:
python scripts/evaluate_retrieval.py --no-justification
```
