# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time  
**Team:** Corbin Christiansen, Jun Kim, Eli Barnes  
**Date:** Nov 5, 2026

## Data

- Where did the data come from, and how did you obtain or prepare them?  
  - The data come from Kaggle's Titanic: Machine Learning from Disaster competition. We downloaded `train.csv` (891 labeled passengers) and `test.csv` (418 unlabeled passengers) into `data/`. Each row is one passenger. We have not cleaned or transformed the data yet; the baseline only selects the numeric columns. 
- What data are available to your team now? What remains to be collected or cleaned?  
  - All competition data are available, so nothing remains to be collected. Cleaning has not started.
- **Route C:** What does your first collected and cleaned sample contain?  
  - Not applicable; we chose Route B. 


## Evaluation plan
None of the following has been implemented yet; the baseline uses a single 80/20 split.

- What comparison, split, metric, or diagnostic will help answer your question? Why?
  - We plan to use repeated stratified 5-fold cross-validation with the same folds for every method: the sex-only rule, Pipeline 1 (passenger-level features), and Pipeline 2 (adding ticket and surname group features). We will report accuracy, the Kaggle metric, and log loss, which shows whether predicted probabilities improve. We will compare methods by fold-by-fold differences, because with 891 passengers a single split is too noisy to separate small differences. As a stability check, we will also hold out whole ticket groups.
- How will you avoid data leakage or another misleading comparison?
  - We will fit imputation and encoding inside each training fold. Group survival features will use training-fold labels only and exclude the passenger's own label. We will drop identifiers such as `PassengerId`, compare methods only on identical folds, and not tune on the Kaggle leaderboard.
- **Route B:** How will you validate locally before making a Kaggle submission?
  - We will cross-validate each candidate and record its command, configuration, commit, and scores in `results/runs.csv` before submitting. With 418 test passengers, the Kaggle score's standard error is about 2 percentage points, so we will not treat smaller differences as meaningful.

## First runnable result

- What baseline or other reference did you run? What did it produce?
  - We ran a simple Random Forest Classifier on just the numeric data from the dataset. This gives us a great baseline as to what a simple ML model should be able to do. Our model produced a train/test accuracy of 100%/76.54% respectively.
- Where are the code and output? Link them, and include the command needed to rerun the result if available.
  - The code is included in our [Project Baseline](project_baseline.ipynb)
- What did this first result suggest, and what will you try next?
  - This suggests that simple numeric features do great on the training data, but will need to do a lot of tuning in order to fully model the unseen data.
