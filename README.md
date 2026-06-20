# Practical Application III: Comparing Classifiers
## Bank Telemarketing Campaign — Predicting Term Deposit Subscriptions

**[View the Jupyter Notebook →](bank_classifier_comparison.ipynb)**

---

## Summary of Findings

### Business Problem
A Portuguese bank runs telephone marketing campaigns to sell term deposit products. With only ~11% of clients subscribing, the bank needs to identify the most promising prospects to reduce wasted calls and improve ROI.

### Dataset
- **Source:** UCI Machine Learning Repository (Moro et al., 2014)
- **Size:** 41,188 client contacts from 17 campaigns (May 2008 – November 2010)
- **Task:** Binary classification — predict whether a client will subscribe (`yes`/`no`)
- **Key note:** `duration` excluded (data leakage — call length is unknown before calling)

### Models Compared

| Model | Test Accuracy | ROC-AUC (Tuned) |
|-------|:---:|:---:|
| Logistic Regression ✅ **Best** | ~90.2% | ~0.79 |
| K-Nearest Neighbors | ~90.1% | ~0.78 |
| Decision Tree | ~90.1% | ~0.78 |
| SVM | ~89.1% | ~0.77 |

**Evaluation Metric:** ROC-AUC was chosen over accuracy because the dataset is class-imbalanced (89% "no", 11% "yes"). A naive model predicting "no" always achieves 89% accuracy but catches zero subscribers.

### Key Actionable Insights

1. **Timing:** Campaigns in March, September, October, and December yield the highest subscription rates
2. **Economic conditions:** Lower Euribor rates and negative employment variation → higher subscription willingness
3. **Previous success:** Clients who subscribed in a past campaign are 4× more likely to subscribe again
4. **Contact method:** Cellular contacts significantly outperform landline contacts
5. **Don't over-contact:** Subscription rate drops after 2–3 attempts; cap contacts at 3 per campaign

### Recommendation
Deploy **Logistic Regression** (tuned C=0.1) to score all customers before each campaign. Call only the top 30% by predicted probability — this maintains ~70% of conversions while cutting call volume by more than half.

### Next Steps
- Explore ensemble methods (XGBoost, Random Forest) for potentially higher AUC
- Address class imbalance via SMOTE or class-weight adjustments
- Monitor model drift quarterly as macroeconomic conditions change

---

## Project Structure

```
├── README.md
├── bank_classifier_comparison.ipynb   ← Main analysis notebook
├── bank-additional-full.csv           ← Dataset (41,188 records)
├── bank-additional.csv                ← 10% sample (for SVM prototyping)
└── CRISP-DM-BANK.pdf                  ← Accompanying research paper
```

## Citation
Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
