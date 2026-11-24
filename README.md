# Data Storm 6.0: predicting insurance agents at risk of a zero-sales month

Our entry to the Data Storm 6.0 data science competition (May 2025), as team Calm Coders (Chandupa Ambepitiya
and Sithum De Zoysa). The data has one row per insurance agent per month: proposals, quotations, customers
contacted, policies sold, premium value, income and more. The tasks were to predict which agents will sell no
policies next month ("NILL" agents), explain why, and suggest how to help them.

## Approach

1. **Exploratory analysis:** distributions and outliers, monthly sales in 2023 and 2024, and correlations.
   We also derived new measures, such as a behavioural consistency score and the ramp-up time of new agents.
2. **Segmentation:** K-Means on per-agent metrics splits the agents into high, medium and low performers; the
   report follows how each group develops over time.
3. **Prediction:** the target is whether an agent sells at least one policy in the following month (each
   agent's records shifted by one month, with leaking columns removed). An XGBoost classifier was tuned with
   Optuna (learning rate, depth, number of trees, subsampling, regularisation) to maximise F1 under 5-fold
   stratified cross-validation. **F1 = 0.945** on the competition's Kaggle leaderboard.
4. **Explanations and action plans:** agents whose predicted probability of selling is below 0.7 are flagged
   as at risk. For each of them, SHAP values pick the three features that pushed the risk up most, and each
   feature is mapped to a coaching strategy. The output is a per-agent table of risk score, reasons and plan.

The features with the highest XGBoost gain were ANBP value, the number of cash-payment policies, net income,
the number of policyholders and unique quotations.

## Files

| File | Contents |
|---|---|
| `EDA.ipynb`, `EDA1.ipynb` | Exploratory analysis (`EDA1` is the extended version) |
| `Part2.ipynb` | Per-agent metrics and K-Means segmentation |
| `predictions.ipynb` | Target construction, Optuna tuning, the final XGBoost model and test-month predictions |
| `personalized_action_plans.ipynb` | At-risk flagging, SHAP explanations and coaching plans |
| `Calm_Coders_report.pdf` | Our report (28 pages) |

The competition data is not included.
