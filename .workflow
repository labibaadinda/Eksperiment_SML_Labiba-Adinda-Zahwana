# stages:
#   # Step 1: Data Preprocessing
#   preprocess:
#     cmd: |
#       python preprocess_data.py
#     deps:
#       - data/raw_data.csv
#       - src/preprocessing_script.py
#     outs:
#       - data/processed_data.csv
#     params:
#       - preprocessing_config.yaml

#   # Step 2: Train-Test Split
#   split_data:
#     cmd: |
#       python split_data.py
#     deps:
#       - data/processed_data.csv
#       - src/split_data_script.py
#     outs:
#       - data/train_data.csv
#       - data/test_data.csv

#   # Step 3: Model Training (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
#   train_models:
#     cmd: |
#       python train_models.py
#     deps:
#       - data/train_data.csv
#       - src/train_model_script.py
#     outs:
#       - models/logistic_regression.pkl
#       - models/random_forest.pkl
#       - models/gradient_boosting.pkl
#       - models/xgboost.pkl
#     params:
#       - model_config.yaml

#   # Step 4: Model Evaluation (Evaluating performance on test data)
#   evaluate_models:
#     cmd: |
#       python evaluate_models.py
#     deps:
#       - data/test_data.csv
#       - models/logistic_regression.pkl
#       - models/random_forest.pkl
#       - models/gradient_boosting.pkl
#       - models/xgboost.pkl
#       - src/evaluate_model_script.py
#     outs:
#       - results/accuracy_scores.csv
#     params:
#       - evaluation_config.yaml

#   # Step 5: Save Processed Data & Final Model Results
#   save_results:
#     cmd: |
#       python save_results.py
#     deps:
#       - results/accuracy_scores.csv
#       - models/xgboost.pkl
#     outs:
#       - results/final_results.csv
#       - models/final_xgboost_model.pkl
#     params:
#       - result_config.yaml

#   # Step 6: Visualization (Correlation Matrix and Distribution Plots)
#   visualize:
#     cmd: |
#       python visualize_data.py
#     deps:
#       - data/processed_data.csv
#       - src/visualization_script.py
#     outs:
#       - results/correlation_matrix.png
#       - results/distribution_plots.png
#     params:
#       - visualization_config.yaml

#   # Step 7: Model Deployment (Optional: Save the final model for deployment)
#   deploy_model:
#     cmd: |
#       python deploy_model.py
#     deps:
#       - models/final_xgboost_model.pkl
#     outs:
#       - models/deployed_model.pkl
#     params:
#       - deployment_config.yaml
