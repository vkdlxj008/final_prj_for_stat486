# Working Investigation

**Checkpoint:** November 12, 2026, by 11:59 PM Mountain Time  
**Team:** Corbin Christiansen, Jun Kim, Eli Barnes  
**Route:** B
**Date:** 11/12/2026

Use the prompts for your chosen route to describe what is working now and what you learned from a first check. Add a new, dated entry for the working investigation to the root `README.md` of your team's GitHub repository and link the relevant code and output there. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; it does not replace the README entry. Preliminary or negative results are welcome when you explain what they show.

## Working code and evidence

- What part of your investigation runs now? Link the code, configuration, and an output it produced.

The entire baseline pipeline executes successfully—from loading raw competition data (`train.csv` and `test.csv`), through feature preparation and preprocessing using Scikit-Learn transformers, to local cross-validation evaluation and final inference.

- What inputs and steps does the run use? Give the command to reproduce the first check if available, and note any data access or setup needed.

The pipeline is executed inside [`project_baseline.ipynb`](https://github.com/vkdlxj008/final_prj_for_stat486/blob/main/project_baseline.ipynb). It ingests raw CSV inputs from the `data/` directory, applies imputation and scaling, fits a baseline `RandomForestClassifier`, and outputs evaluation metrics.

- What comparison or diagnostic did you run, using which evaluation split, metric, or quality measure? What does the result show so far?

We ran 5-fold cross-validation evaluating out-of-fold accuracy on the training split to establish our baseline performance before introducing feature engineering (such as travel companion features).

## Route-specific progress

Complete the prompts for your chosen route.

### Route B: Kaggle competition

- Where is the working path from raw competition data through preparation, local validation, training, and inference to a submission file? Link the code and record the run command or configuration.

The complete pipeline is defined sequentially in [`project_baseline.ipynb`](https://github.com/vkdlxj008/final_prj_for_stat486/blob/main/project_baseline.ipynb), concluding with the test-set transformation and export script that generates `submission.csv`.

- What was the local validation result, and why is that validation appropriate for the competition?

Using 5-fold cross-validation on the 891 training samples, our baseline random forest achieved a stable out-of-fold accuracy score. This is appropriate because it prevents overfitting to a single train-test split and closely mirrors Kaggle's unseen test distribution.

- Link the first **valid Kaggle submission** and record its Kaggle result. Which pipeline and commit produced the submission file? What will you change or test for the second pipeline?

[`kaggle results`](https://www.kaggle.com/competitions/titanic/submissions) 

This is our first submission to kaggle. We got a score of 0.78708. For our second pipeline, we are going to try and use a gradient boost.

## Next question

- If your plan changed since November 5, what evidence or feedback led to the change?

We have no intent to change plans.

- What is uncertain or blocked now, and how will you adjust the scope if needed?

We need some better way to gain test accuracy. If the second pipeline does not help our predictive power, we will need to come up with an alternative course of action to solve that issue.
