# P7Q5 search-policy replay and analysis

Frozen before revealing policy outcomes from the completed timing labels.

- Input contract SHA-256: `dbed44f6260f6acade9dd723875f86f53b843402603447c4a927b7fb618f2246`.
- Timing inputs are the four completed P7Q1--P7Q4 workload summaries.
- B3 uses the already frozen ConfigEntity-only XGBoost implementation, seed, and hyperparameters.
- Compare gross budgets 4, 8, 16, and 24; the invalid W04 candidate consumes a gross dispatch if encountered and supplies no label.
- Primary outcomes: best valid median latency seen, regret to the physically measured valid-pool oracle, oracle hit rate, and gross dispatch position of the oracle.
- B2 is summarized over the 1000 deterministic random permutations fixed by the original contract.
- No further board measurements or candidate replacement are allowed during analysis.
