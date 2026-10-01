# results/opt — TellTail-OPT paper artifacts

Golden artifacts for the optimization-based attack, at paper scale (the demo under
`demo/` is separate and smaller).

```
optimized_queries/
  tm1_tm2.csv   optimized queries for TM1 and TM2 (passage-level triggers; the two
                threat models share the same queries — only the scoring hit rule differs)
  tm3.csv       optimized queries for TM3 (topic-level / centroid triggers)
responses/
  <eval_model>.jsonl   TM3 RAG responses (gpt-4o-mini), one file per evaluated retriever,
                       as generated before judging. LLM judgments will be added here.
```

Scores (OSCR curves, QB) live under `results/paper/`.
