# Evidence and Interpretation

**Checkpoint:** November 19, 2026, by 11:59 PM Mountain Time  
**Team:** Corbin Christiansen, Jun Kim, Eli Barnes
**Route:** B  
**Date:** 11/19/2026

Add a new, dated entry to the root `README.md` of your team's GitHub repository. Link the code, results, and figures or tables that support your summary. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; preliminary or negative results are welcome.

## Evidence so far

- **Route B:** Compare the pipelines using local validation and any Kaggle results available. What substantive design change did you make, and did results improve or degrade? Link each result to its pipeline and run configuration.

  By adding Gradient Boosting we did better than the Grid search and our Kaggle results, getting our training accuracy up to **91%** and our test accuracy up to **84%**. This was somehow still worse than our baseline model of a simple Random Forrest Classifier.
  Supporting code can be found [here](project_baseline.ipynb)

## Interpretation and next step

- Describe one informative error, failure, tradeoff, or limitation. How does it affect your conclusion?

  Both models fit the training data substantially better than their test accuracy suggests: Gradient Boosting was 91.71% train versus 83.80% test accuracy, while Random Forest was 98.31% versus 84.36%. This gap, especially for RF, is evidence of overfitting.

- What did the evidence change in your thinking, and what will you test or revise next?

  The results make the model choice look less decisive than a single accuracy number suggests: both ensembles perform similarly on the held-out split, and the high training scores flag variance/overfitting. For future models we will look to stop this overfitting by using more regularzation methods to keep our models less complex such as feature selection.
