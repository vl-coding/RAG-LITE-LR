# Retrieval Evaluation

Results from running `scripts/evaluate_retrieval.py` against the 17 gold
queries in `tests/eval/gold_queries.yaml` (top-5, HyDE + Claude
justification enabled, `claude-haiku-4-5`). Full per-query detail
(retrieved IDs, justifier scores) is written to `outputs/eval_report.json`
each time the script runs.

## Headline metrics (mean across 17 queries, top-5)

| Metric | Value |
|---|---|
| Precision@5 | 0.647 |
| Recall@5 | 0.504 |
| NDCG@5 | 0.721 |
| MRR | 0.882 |

An MRR of 0.88 means the first gold-relevant result lands at rank 1 or 2
for nearly every query. Precision@5 of ~0.65 means roughly 3 of the top 5
results are gold-relevant on average.

## Per-query results

| Query | P@5 | R@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| What programs help improve early literacy outcomes for low-income preschoolers? | 0.400 | 0.500 | 0.414 | 0.500 |
| How big a problem is summer learning loss for low-income students, and what can schools do about it? | 0.800 | 1.000 | 1.000 | 1.000 |
| What interventions actually reduce chronic absenteeism in K-12 schools? | 0.600 | 0.300 | 0.723 | 1.000 |
| What does the research say about school-based mental health programs and their effectiveness? | 0.200 | 0.250 | 0.246 | 0.500 |
| What does trauma-informed practice look like in schools and does it help students? | 0.800 | 0.444 | 0.854 | 1.000 |
| How can we help first-generation students get into and succeed in college? | 0.400 | 0.500 | 0.637 | 1.000 |
| What family and parent engagement strategies improve outcomes for students? | 0.800 | 0.500 | 0.830 | 1.000 |
| Do after-school programs improve academic outcomes for at-risk students? | 1.000 | 0.714 | 1.000 | 1.000 |
| What approaches help local communities participate effectively in conservation programs? | 0.800 | 0.571 | 0.854 | 1.000 |
| How are local communities adapting to climate change, and what strategies are working? | 0.400 | 0.500 | 0.637 | 1.000 |
| What does community-based participatory research look like in practice? | 1.000 | 0.625 | 1.000 | 1.000 |
| How can organizations evaluate the impact of their community health programs? | 0.400 | 0.250 | 0.553 | 1.000 |
| How can arts-based programs support youth development and community engagement? | 0.600 | 0.375 | 0.530 | 0.500 |
| What strategies help preserve and promote cultural heritage in local communities? | 0.800 | 0.571 | 0.869 | 1.000 |
| What does the research say about environmental health disparities in low-income or marginalized communities? | 1.000 | 0.556 | 1.000 | 1.000 |
| How do nonprofit organizations measure and improve their own effectiveness? | 0.800 | 0.667 | 0.869 | 1.000 |
| How can creative arts programs support young people's mental health and wellbeing? | 0.200 | 0.250 | 0.246 | 0.500 |

**Weakest queries:** the two mental-health-focused queries ("school-based
mental health programs" and "creative arts programs ... mental health and
wellbeing") both scored P@5=0.20 / NDCG@5=0.246, retrieving only one of
their gold-relevant documents each. Both also share a gold doc
(`openalex:W4390519162`), suggesting the corpus may simply have limited
coverage of this overlap area, or the dense/BM25 candidates need a larger
pool for these topics.

## Latency

| Metric | Value |
|---|---|
| Mean total | 5.56s |
| Min | 4.44s |
| Max | 8.49s |

Dense (~0.03s) and BM25 (~0.001s) retrieval are negligible; nearly all
latency comes from the Claude HyDE and justification calls.

## Justifier score calibration

| Score | n | Mean | Stdev | Min | Max |
|---|---|---|---|---|---|
| relevance_score | 85 | 8.91 | 1.04 | 5 | 10 |
| specificity_score | 85 | 7.69 | 1.31 | 4 | 10 |

Scores skew high and cluster tightly, which is expected for an
already-filtered top-5 set, but leaves limited headroom for the justifier
to discriminate between borderline results.

## Reproducing

```bash
python scripts/evaluate_retrieval.py
# or, to skip Claude justification calls:
python scripts/evaluate_retrieval.py --no-justification
```
