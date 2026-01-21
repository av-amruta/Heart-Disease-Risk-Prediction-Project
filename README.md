# Heart Disease Risk Prediction Project

## a. Problem Statement
The goal of this project is to predict the risk of heart disease or myocardial infarction using high-impact health indicators from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset. Given the critical nature of medical diagnosis, the project places a strong emphasis on Recall to minimize false negatives, ensuring that potential cases of heart disease are not overlooked during screening.

## b. Dataset Description
- **Source:** Kaggle - [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)

- **Size:** The dataset contains 253,680 records with 22 initial health indicators.

- **Features:** The final models utilize 16 selected high-impact features as listed below:<br>

    | Health Indicators | Lifestyle & Demographics |
    |------------------|-------------------------|
    | HighBP           | Smoker                  |
    | HighChol         | PhysActivity            |
    | BMI              | Fruits                  |
    | Stroke           | Veggies                 |
    | Diabetes         | HvyAlcoholConsump       |
    | GenHlth          | Age                     |
    | MentHlth         | Sex                     |
    | PhysHlth         | DiffWalk                |

- **Target Variable:** **HeartDiseaseorAttack** (Binary: 0 for no disease, 1 for heart disease/attack).

- **Class Distribution:** The dataset is highly imbalanced, with approximately 90.6% of participants reporting no heart disease and 9.4% reporting heart disease or attack.

## c. Models Used
The following six machine learning models were trained and evaluated.

**Comparison Table: Validation Evaluation Metrics**<br>
*Note: Metrics are based on the validation set evaluation*

| ML Model Name          | Accuracy | AUC    | Precision | Recall  | F1     | MCC    |
|-----------------------|---------|--------|-----------|--------|-------|-------|
| Logistic Regression    | 0.756   | 0.846  | 0.250     | 0.796  | 0.380 | 0.349 |
| Decision Tree          | 0.725   | 0.819  | 0.227     | 0.797  | 0.353 | 0.320 |
| K-Nearest Neighbor     | 0.709   | 0.786  | 0.209     | 0.753  | 0.328 | 0.282 |
| Naive Bayes            | 0.788   | 0.807  | 0.246     | 0.607  | 0.350 | 0.286 |
| Random Forest          | 0.753   | 0.843  | 0.246     | 0.787  | 0.375 | 0.342 |
| XGBoost                | 0.745   | 0.847  | 0.243     | 0.808  | 0.374 | 0.344 |

## d. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression is good at detecting positives (high recall) but generates many false positives (low precision). It has highest MCC. It’s balanced overall but may overpredict positives. |
| Decision Tree | Decision Tree overpredicts positives and has a bit worse overall discrimination than Logistic Regression. It’s not very precise, though it captures most positives. |
| kNN | KNN performs the worst overall, especially in distinguishing classes (lowest AUC) and precision. Good recall isn’t enough to compensate for poor precision. |
| Naive Bayes | Naive Bayes sacrifices recall for higher accuracy. It predicts positives more conservatively, missing many true positives. Could be preferred if false positives are costly. |
| Random Forest (Ensemble) | Random Forest balances high recall with decent accuracy. Slightly lower than Logistic Regression in precision/F1 making it a decent model ranked below logistic regression. |
| XGBoost (Ensemble) | **Top Performer:** XGBoost excels at detecting positives (highest recall) which is most important for disease detection and has the best AUC, showing strong ranking ability. Precision is low, so it tends to overpredict positives, but it’s one of the stronger models in terms of Recall. |

## e. Folder Structure
```plaintext
project-root/
│
├── data/
│   ├── heart_disease_health_indicators_BRFSS2015.csv  # Dataset with health indicators
│   └── heart_disease_raw_test_set.csv                 # Raw test dataset
│
├── metrics/
│   └── detailed_metrics.pkl                           # Stored model validation evaluation metrics
│
├── models/
│   ├── scaler/
│   │   └── scaler.pkl                             # Scaler object for feature scaling
│   ├── decision_tree.pkl                          # Decision Tree model
│   ├── knn.pkl                                    # k-Nearest Neighbors model
│   ├── logistic_regression.pkl                    # Logistic Regression model
│   ├── naive_bayes.pkl                            # Naive Bayes model
│   ├── random_forest.pkl                          # Random Forest model
│   └── xgboost.pkl                                # XGBoost model
│
├── app.py                                        # Streamlit application script
├── config.json                                   # Configuration file with paths and features
├── model building.ipynb                          # Jupyter notebook for model building
├── README.md                                     # Project README file
└── requirements.txt                              # Python dependencies list
```

## f. Setup Instructions
To run the app locally, follow these steps:

1. Clone the GIT repo
2. Create and activate a virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Run the app with `streamlit run app.py`

## g. References

- Kaggle Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- XGBoost Documentation: https://xgboost.readthedocs.io/

