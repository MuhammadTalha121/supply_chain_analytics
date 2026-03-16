# Supply Chain Analytics & Cost Prediction

## Problem
Supply chain operations generate massive data across inventory, logistics, manufacturing, and sales — yet most businesses can't predict costs or identify inefficiencies until it's too late. This project answers: **what drives supply chain costs, and can we predict them?**

## Approach
End-to-end analysis on real supply chain data (100 SKUs, 24 variables) covering product sales, stock levels, lead times, defect rates, transportation modes, and manufacturing costs.

**Why this approach:**
- Started with EDA to understand data structure and spot patterns
- Identified key cost drivers through correlation and feature analysis
- Built and compared 7 ML models (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, SVR, MLP) with cross-validation
- Applied GridSearchCV hyperparameter tuning on top performers

## Results
| Model | Best CV R² |
|-------|-----------|
| SVR | -0.031 |
| Random Forest | -0.199 |
| Gradient Boosting | -0.613 |

SVR performed best. Negative R² values indicate high cost variability — a finding itself, suggesting external factors (demand spikes, route disruptions) dominate cost behavior beyond structured features alone.

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`

## Key Insights
- Transportation mode and route significantly impact total costs
- Defect rates vary by product type — haircare vs skincare show different quality patterns
- Lead time and manufacturing cost are weakly correlated with final cost, suggesting logistics is the dominant cost driver

## Dataset
Supply chain dataset with 24 features including SKU, pricing, stock levels, lead times, production volumes, inspection results, defect rates, and shipping routes.
