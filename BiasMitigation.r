install.packages(c(
  "tidymodels",   # Fair model building & reweighting
  "DALEX",        # Explainability & disparity exploration
  "fairmodels",   # Fairness metrics & parity evaluation
  "tidyverse",    # Data manipulation
  "recipes",      # Preprocessing HR data
  "themis",       # Handling class imbalance
  "yardstick"     # Model evaluation
))

# Exploring the dataset
library(tidyverse)
# Load the small dataset first
df <- read_csv("~/Downloads/data.csv")
# Explore it
glimpse(df)
colnames(df)
head(df)

# STEP 1 - CLEANING AND PREPARING DATA
# Convert target and protected attribute to factors
df <- df %>%
  mutate(
    HiringDecision     = as.factor(HiringDecision),
    Gender             = factor(Gender, levels = c(0,1), labels = c("Female","Male")),
    EducationLevel     = as.factor(EducationLevel),
    RecruitmentStrategy = as.factor(RecruitmentStrategy)
  )
# Check class balance
table(df$HiringDecision)
table(df$Gender)

# STEP 2 - Visualize Bias (EDA)
# By Gender
ggplot(df, aes(x = Gender, fill = HiringDecision)) +
  geom_bar(position = "fill") +
  labs(title = "Hiring Rate by Gender", y = "Proportion", x = "Gender") +
  scale_fill_manual(values = c("#e74c3c","#2ecc71"), labels = c("Not Hired","Hired")) +
  theme_minimal()
# By Age group
df %>%
  mutate(AgeGroup = cut(Age, breaks = c(20,30,40,50,60))) %>%
  ggplot(aes(x = AgeGroup, fill = HiringDecision)) +
  geom_bar(position = "fill") +
  labs(title = "Hiring Rate by Age Group", y = "Proportion") +
  scale_fill_manual(values = c("#e74c3c","#2ecc71"), labels = c("Not Hired","Hired")) +
  theme_minimal()

# STEP 3 - Splitting Data for Modeling
library(tidymodels)
set.seed(42)
split <- initial_split(df, prop = 0.8, strata = HiringDecision)
train <- training(split)
test  <- testing(split)
cat("Train rows:", nrow(train), "\nTest rows:", nrow(test))

# STEP 4 - Preprocessing
hr_recipe <- recipe(HiringDecision ~ ., data = train) %>%
  step_normalize(Age, ExperienceYears, DistanceFromCompany,
                 InterviewScore, SkillScore, PersonalityScore) %>%
  step_dummy(Gender, EducationLevel, RecruitmentStrategy) %>%
  step_zv(all_predictors())
summary(hr_recipe)

# STEP 5 - Train the Model
install.packages("ranger")
library(ranger)
# Define Random Forest model
rf_spec <- rand_forest(trees = 100) %>%
  set_engine("ranger", probability = TRUE) %>%
  set_mode("classification")
rf_wf <- workflow() %>%
  add_recipe(hr_recipe) %>%
  add_model(rf_spec)
# Train
rf_fit <- fit(rf_wf, data = train)
cat("Model trained successfully!")

# STEP 6 - Check Model Accuracy
# Predicting on test set
predictions <- predict(rf_fit, test, type = "class") %>%
  bind_cols(predict(rf_fit, test, type = "prob")) %>%
  bind_cols(test)
# Accuracy metrics
metrics <- predictions %>%
  metrics(truth = HiringDecision, estimate = .pred_class)
print(metrics)
# Confusion matrix
conf_mat(predictions, truth = HiringDecision, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

# STEP 7 - DALEX
install.packages("DALEXtra")
library(DALEXtra)
# Creating explainer
explainer <- explain_tidymodels(
  rf_fit,
  data  = test %>% select(-HiringDecision),
  y     = as.numeric(test$HiringDecision) - 1,
  label = "RF Baseline",
  verbose = FALSE
)
# Variable importance 
vi <- model_parts(explainer)
plot(vi) + 
  ggtitle("What factors drive hiring decisions?")

# STEP 8 - Fairness Check
library(fairmodels)
# Running fairness on Gender
fobject <- fairness_check(
  explainer,
  protected  = test$Gender,
  privileged = "Male",
  label      = "RF Baseline"
)
# Printing summary
print(fobject)
# Plotting fairness metrics
plot(fobject)

# STEP 9 - Mitigating Bias by Reweighting
weights <- reweight(
  protected = train$Gender,
  y = as.numeric(as.character(train$HiringDecision))
)
summary(weights)

# STEP 10 - Retraining Model 
library(tidymodels)
# Extracting preprocessed training data
train_baked <- hr_recipe %>%
  prep() %>%
  bake(new_data = train)
# Defining Fair Model
rf_spec_fair <- rand_forest(trees = 100) %>%
  set_engine("ranger", 
             probability = TRUE,
             case.weights = weights) %>%
  set_mode("classification")
# Building workflow
rf_wf_fair <- workflow() %>%
  add_recipe(hr_recipe) %>%
  add_model(rf_spec_fair)
# Training
rf_fit_fair <- fit(rf_wf_fair, data = train)
cat("Model trained successfully!")

# STEP 11 - Explaining Fair Model
explainer_fair <- explain_tidymodels(
  rf_fit_fair,
  data    = test %>% select(-HiringDecision),
  y       = as.numeric(test$HiringDecision) - 1,
  label   = "RF Fair",
  verbose = FALSE
)
cat("Fair explainer created!")

# STEP 12 - Comparing Baseline vs Fair Model
fobject_both <- fairness_check(
  explainer,       # baseline (biased)
  explainer_fair,  # fair model
  protected  = test$Gender,
  privileged = "Male"
)

print(fobject_both)
plot(fobject_both)

# STEP 13 — Accuracy Comparison
preds_fair <- predict(rf_fit_fair, test, type = "class") %>%
  bind_cols(test)
cat("Baseline Model\n")
predictions %>%
  metrics(truth = HiringDecision, estimate = .pred_class) %>%
  print()
cat("\nFair Model\n")
preds_fair %>%
  metrics(truth = HiringDecision, estimate = .pred_class) %>%
  print()

# STEP 14 — Generating Final Bias Report
library(patchwork)
# Plot 1: Hiring rate by gender
p1 <- ggplot(df, aes(x = Gender, fill = HiringDecision)) +
  geom_bar(position = "fill") +
  labs(title = "1. Hiring Rate by Gender",
       y = "Proportion", x = "Gender") +
  scale_fill_manual(values = c("#e74c3c","#2ecc71"),
                    labels = c("Not Hired","Hired")) +
  theme_minimal()
# Plot 2 - Variable importance
p2 <- plot(model_parts(explainer)) +
  ggtitle("2. Baseline - Key Hiring Factors")
# Plot 3 - Variable importance fair model  
p3 <- plot(model_parts(explainer_fair)) +
  ggtitle("3. Fair Model - Key Hiring Factors")
# Plot 4 - Fairness comparison
p4 <- plot(fobject_both) +
  ggtitle("4. Fairness Comparison: Baseline vs Fair")
# Print
print(p1)
print(p2)
print(p3)
print(p4)

