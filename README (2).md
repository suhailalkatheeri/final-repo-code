# Bias Mitigation Pipeline in AI-Driven Hiring Processes

# Project Overview

This project builds an end-to-end fairness-aware machine learning pipeline using open-source R packages. It trains a baseline Random Forest classifier on a hiring dataset, audits it for gender bias using DALEX and fairmodels, applies sample reweighting to mitigate bias, and compares the fair model against the baseline across both accuracy and fairness dimensions.

# Dataset

- **Source:** Kaggle — [Bias Detection in AI Hiring Decisions](https://www.kaggle.com/datasets/amrsaid1222/data-csv)
- **Rows:** 1,500 synthetic candidate records
- **Features:** 11 (Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies, DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore, RecruitmentStrategy, HiringDecision)
- **Target:** HiringDecision (0 = Not Hired, 1 = Hired)
- **Protected Attribute:** Gender (0 = Female, 1 = Male)

# Requirements

- R version 4.5.1
- RStudio

## R Packages

| Package | Version | Purpose |
|---------|---------|---------|
| tidyverse | 2.0.0 | Data manipulation and visualization |
| tidymodels | 1.4.1 | Model building, recipes, workflows |
| ranger | 0.18.0 | Random Forest engine |
| DALEX | 2.5.3 | Model explainability |
| DALEXtra | 2.3.1 | tidymodels + DALEX integration |
| fairmodels | 1.2.2 | Fairness metrics and bias mitigation |
| recipes | 1.3.2 | Preprocessing |
| yardstick | 1.4.0 | Model evaluation metrics |

Install all packages with:

```r
install.packages(c(
  "tidyverse", "tidymodels", "ranger",
  "DALEX", "DALEXtra", "fairmodels",
  "recipes", "yardstick", "themis"
))
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load and clean dataset |
| 2 | Exploratory Data Analysis (gender & age bias visualizations) |
| 3 | Train/test split (80/20, stratified) |
| 4 | Preprocessing recipe (normalize, encode, remove zero-variance) |
| 5 | Train baseline Random Forest model |
| 6 | Evaluate accuracy (confusion matrix, kappa) |
| 7 | Create DALEX explainer and variable importance plot |
| 8 | Fairness audit using fairmodels (13 metrics, Gender as protected attribute) |
| 9 | Compute sample weights using reweight() |
| 10 | Retrain fair model with case weights |
| 11 | Create DALEX explainer for fair model |
| 12 | Compare baseline vs fair model on fairness metrics |
| 13 | Compare accuracy across both models |

---

## Results

| Metric | Baseline RF | Fair RF |
|--------|------------|---------|
| Accuracy | 90.7% | 90.7% |
| Cohen's Kappa | 0.771 | 0.772 |
| Fairness Metrics Passed | 5 / 5 | 4 / 5 |
| Total Fairness Loss | 0.1707 | 0.4676 |


---

## How to Run

1. Clone or download this repository
2. Place `data.csv` in the project root directory
3. Open `bias_pipeline.R` in RStudio
4. Run all steps using **Cmd + A** then **Cmd + Enter** (Mac) or **Ctrl + A** then **Ctrl + Enter** (Windows)

