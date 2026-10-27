# results/

Everything the report (`docs/RESULTS.md`) is built on is listed here. Tables are generated from these files by `scripts/make_tables.py`.

| files | what | settings |
|---|---|---|
| `final_f1_test.*` | all methods on the 207 test routes (lambda, rho, mu variants, zonal LNS/SA, global SA) | `scripts/run_final.sh` F1 |
| `final_f2_ortools.*` | OR-Tools on the 207 test routes, run on an otherwise idle machine | F2 |
| `final_f3_literature.*` | the 1,107 High+Delivered routes of the LKH-AMZ paper (closed tours, official score) | F3 |
| `final_f4_timing.*` | clean single-process timings on 20 test routes | F4 |
| `final_f5_gnn_order.*` | zonal methods with the GNN(+reference) zone order, 207 test routes | -- |
| `exp13_ortools.*` | first OR-Tools run (lambda = 0 rows only are used, as "run 1") | -- |
| `exp40_solver_study_*.csv` | CP-SAT vs MIP vs LNS on small and full-size routes | `scripts/solver_study.py` |
| `zone_gnn_ref.*`, `zone_gnn_geom.*` | zone-order GNN with / without the reference-route features (`.pt` weights are git-ignored) | `scripts/train_zone_gnn.py` |
| `tuning/t1,t2,t3,t6,t7` | parameter tuning on 60 VALIDATION routes (lambda = 0 rows) | -- |
| `gnn*.history.json` | training histories of the route-level GNN (first branch) | -- |
| `figures/` | figures made by `scripts/make_figures.py` | -- |

`archive_before_zone_fix/` and `tuning/archive_before_zone_fix/` hold runs made **before** stops without a zone ID were fixed (they were treated as the
depot's zone, which distorted every zone-dependent number). They are kept only for the record. Do not use them.
