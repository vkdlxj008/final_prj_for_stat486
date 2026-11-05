# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time  
**Team:** Corbin Christiansen, Jun Kim, Eli Barnes  
**Date:** Nov 5, 2026

## Data

- Where did the data come from, and how did you obtain or prepare them?
- What data are available to your team now? What remains to be collected or cleaned?
- **Route C:** What does your first collected and cleaned sample contain?

## Evaluation plan

- What comparison, split, metric, or diagnostic will help answer your question? Why?
- How will you avoid data leakage or another misleading comparison?
- **Route B:** How will you validate locally before making a Kaggle submission?

## First runnable result

- What baseline or other reference did you run? What did it produce?
  - We ran a simple Random Forest Classifier on just the numeric data from the dataset. This gives us a great baseline as to what a simple ML model should be able to do. Our model produced a train/test accuracy of 100%/76.54% respectively.
- Where are the code and output? Link them, and include the command needed to rerun the result if available.
  - The code is included in our [Project Baseline](project_baseline.ipynb)
- What did this first result suggest, and what will you try next?
    This suggests that simple numeric features do great on the training data, but will need to do a lot of tuning in order to fully model the unseen data.
