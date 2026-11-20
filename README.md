# final_prj_for_stat486

**Team:** Corbin Christiansen, Jun Kim, Eli Barnes  
**Route:** B, Kaggle competition with two meaningfully different pipelines  
**Plan:** [project_plan.md](project_plan.md)

**Question:** Does knowing what happened to a passenger's travel companions improve Titanic survival prediction beyond the passenger's own characteristics?

## Checkpoint summaries

### Nov 5, 2026: Data and reference result

**Status: in progress. No baseline or pipeline results yet.**

#### Instructor feedback

The instructor said the Titanic data are fine but overused, liked the family approach, and suggested going beyond improving prediction, for example with generative sampling, clustering, or detailed EDA.

**Proposed response (not final):** Keep the two-pipeline comparison and add a generative sampling study. It would ask whether a model that generates passengers independently misses survival dependence within travel groups that a group-level model captures. Diffusion models are one candidate, compared with a simpler reference model. We will finalize this design after the Pipeline 1 result.

#### Data

- **Source:** Kaggle's Titanic competition data: 891 labeled training rows and 418 unlabeled test rows. One row is one passenger.
- **Access:** Each team member accepts the competition rules and downloads the data from Kaggle.
- **Known issues:** `Age` and `Cabin` have many missing values. Surnames and tickets are imperfect indicators of who traveled together.
- **Remaining:** Confirm the competition rules and license, add `data/README.md`, and keep raw data out of the repository unless the rules allow it.

#### Evaluation plan

- **Comparison:** Repeated stratified 5-fold cross-validation, using the same folds for every method. Whole ticket groups will also be held out as a stability check.
- **Metrics:** Accuracy, the competition metric, and log loss, which shows whether predicted probabilities improve.
- **Leakage controls:** Fit imputation and encoding inside each training fold. Compute group survival features from training-fold labels only, and exclude the passenger's own label.
- **Local validation before Kaggle:** Record each cross-validation result, along with its command, configuration, and commit, in `results/runs.csv` before submitting. With only 418 test passengers, small Kaggle score differences may not be meaningful.

#### First runnable result

| Method | CV accuracy | CV log loss | Kaggle score |
|---|---|---|---|
| Sex-only reference (`gender_submission.csv`) | TBD | TBD | TBD |
| Pipeline 1 (passenger-level features, gradient-boosted trees) | TBD | TBD | TBD |

- **Code and output:** TBD
- **Rerun command:** TBD
- **What it suggests:** TBD, after the first run.
- **Next:** Run Pipeline 1, then add leakage-safe ticket and surname group features for Pipeline 2 and compare fold-by-fold differences.

### Oct 27, 2026: Project plan (updated Nov 2, 2026)

Chose Route B with Kaggle Titanic. Pipeline 1 uses passenger-level features, and Pipeline 2 adds ticket and surname group features, including the survival rate of known companions. We expect a small improvement, which may be too small to distinguish from noise. See [project_plan.md](project_plan.md).


### Nov 12, 2026: Working investigation

**Status:** Local evaluation framework, sex-only reference, and Pipeline 1 run end to end. First Kaggle submission made from Pipeline 1.

#### What runs now
- **Code:** `src/final_prj_for_stat486/` (`evaluate.py`, `features.py`, `models.py`, `run.py`)
- **Configs:** `configs/sex_only.toml`, `configs/pipeline1.toml`
- **Rerun:**

uv run python -m final_prj_for_stat486.run --config configs/sex_only.toml
uv run python -m final_prj_for_stat486.run --config configs/pipeline1.toml

- **Outputs:** fixed folds in `results/folds_5x5_seed486.csv`, per-fold scores in `results/fold_scores/`, run log in `results/runs.csv`, submissions in `submissions/`

#### Local validation
Repeated stratified 5-fold CV (5 repeats, 25 folds, seed 486), with the same folds for every method. Imputation and encoding are fit inside each training fold. Sex-only probabilities are the training-fold survival rate by sex, so its hard predictions match Kaggle's `gender_submission.csv`.

| Method | Commit | CV accuracy (sd) | CV log loss (sd) | Train acc | Kaggle score |
|---|---|---|---|---|---|
| Sex-only reference | a123cf9 | 0.787 (0.029) | 0.517 (0.037) | 0.787 | TBD |
| Pipeline 1 (RF, passenger features + Title) | a123cf9 | 0.812 (0.031) | 0.801 (0.314) | 0.984 | 0.78708 |

Pipeline 1 features: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title`. Model: `RandomForestClassifier(n_estimators=100, random_state=42)`.

#### What it shows
- **Accuracy:** Pipeline 1 beats sex-only by +2.5 points on average and wins on 20 of 25 folds.
- **Log loss:** Pipeline 1 is much worse (wins on only 4 of 25 folds). The unrestricted RF is overconfident: train accuracy is 98%, and confident wrong predictions inflate log loss. Kaggle scores only accuracy, so this problem is invisible on the leaderboard.

#### Next
- Pipeline 2: add leakage-safe ticket and surname group features. Before that, decide whether to fix RF overconfidence separately (e.g., `min_samples_leaf` or calibration), so the effect of the group features is not confounded with a calibration change.
- Add the ticket-group holdout as a stability check.

### Nov 19, 2026: Evidence and interpretation

**Status:** Gradient Boosting and Random Forest are compared on a held-out 20% split; next work will address overfitting and regularize the models.

#### Route B evidence

The substantive design change was replacing the grid-searched model with Gradient Boosting. The updated comparison is in [the baseline notebook](project_baseline.ipynb), with the 80/20 train/test split and model results recorded there. Gradient Boosting improved on the grid-search result and available Kaggle validation, but it did not improve on the baseline Random Forest's held-out accuracy.

| Method | Train accuracy | Test accuracy | Comparison |
|---|---:|---:|---|
| Grid search / tuned RF | - | 81.56% | Tuned cross-validation accuracy: 79.64% |
| Gradient Boosting | 91.71% | 83.80% | Improved over grid search and the available Kaggle result |
| Baseline Random Forest | 98.31% | 84.36% | 0.56 percentage points above Gradient Boosting on the held-out split |

Supporting outputs: [baseline notebook](project_baseline.ipynb) and [submission history](submissions/pipeline1.csv). The existing Pipeline 1 implementation and run command remain documented above; its configuration is [configs/pipeline1.toml](configs/pipeline1.toml), and the run log is [results/runs.csv](results/runs.csv).

#### Interpretation

Both ensembles fit the training data substantially better than the held-out split: Gradient Boosting had a 7.91-point train/test gap, while Random Forest had a larger 13.95-point gap. This is evidence of variance and overfitting, especially for Random Forest. It also makes model choice less decisive than a single accuracy value suggests: the held-out test accuracies differ by only 0.56 percentage points.

The gradient-boosted model is therefore not a clear winner despite improving training accuracy and beating the grid-search result. We will test regularization and simpler models next, using feature selection and methods such as limiting tree depth, adding minimum leaf requirements, or increasing regularization, while preserving the same folds and comparison structure.
