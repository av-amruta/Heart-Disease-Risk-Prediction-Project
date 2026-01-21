# Import required libraries for web app, data processing, ML models, and visualization
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import json
import os

# Load configuration file containing model paths and feature list
with open("config.json", "r") as f:
    config = json.load(f)

# Base directory of the script
BASE_DIR = os.path.dirname(__file__)

# Paths relative to the repo
model_path = config['paths']['output_paths']['models']
scaler_path = config['paths']['output_paths']['scaler']
val_metrics_path = config['paths']['output_paths']['val_metrics_output']

# Set Streamlit page configuration for the web application
st.set_page_config(page_title="Heart Disease Analysis Portal", layout="wide")

# Load feature list and model path from configuration
SELECTED_FEATURES = config['model_features']

# Cache the model loading to improve app performance and avoid reloading on each interaction
@st.cache_data
def load_assets():
    """Load all trained models, scaler, and metrics from saved pickle files"""
    scaler = joblib.load(scaler_path)
    # Load all six trained models
    models = {
        "Logistic Regression": joblib.load(os.path.join(model_path, 'logistic_regression.pkl')),
        "Decision Tree": joblib.load(os.path.join(model_path, 'decision_tree.pkl')),
        "KNN": joblib.load(os.path.join(model_path, 'knn.pkl')),
        "Naive Bayes": joblib.load(os.path.join(model_path, 'naive_bayes.pkl')),
        "Random Forest": joblib.load(os.path.join(model_path, 'random_forest.pkl')),
        "XGBoost": joblib.load(os.path.join(model_path, 'xgboost.pkl'))
    }
    # Load detailed validation metrics for model analysis
    detailed_metrics = joblib.load(val_metrics_path)
    
    return scaler, models, detailed_metrics

# Load all assets once at startup
scaler, models, detailed_metrics = load_assets()

# Display main page title
st.title("Heart Disease Health Indicators Analysis")
st.markdown("---")

# Create two tabs: one for project overview and one for live model testing
tab1, tab2 = st.tabs(["Project Overview & Analysis", "Live Model Testing"])

with tab1:
    # Tab 1: Display project documentation and model analysis
    st.header("Project Documentation")
    
    # Create two columns for project information
    col1, col2 = st.columns([1, 1.2])
    with col1:
        # Display problem statement
        st.subheader("Problem Statement")
        st.info("Goal: Predict heart disease risk using CDC BRFSS 2015 health indicators. "
                "The project emphasizes **Recall** to minimize false negatives in medical screening.")
        
        # Display dataset information
        st.subheader("Dataset Details")
        st.write("- **Source:** [Kaggle - Heart Disease Health Indicators](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)")
        st.write("- **Features:** 16 high-impact health indicators (e.g., BMI, BP, Age).")

        # Show expandable list of all 16 features
        with st.expander("📋 View All 16 Selected Features"):
            # Format features as bullet points
            features_list = "\n".join([f"- {feat}" for feat in SELECTED_FEATURES])
            st.markdown(features_list)

        # Display class imbalance information
        st.write("- **Classes:** Imbalanced (90% No Disease, 10% Heart Disease).")


    with col2:
        # Display validation metrics in second column
        st.subheader("Validation Evaluation Metrics")

        # Expand section explaining metric definitions
        with st.expander("Metric Glossary & Definitions"):
            st.markdown("""
            | Metric | Scope | Description |
            | :--- | :--- | :--- |
            | **Recall** | Positive Class (1.0) | **Sensitivity**: Percentage of actual Heart Disease cases correctly identified. |
            | **Precision** | Positive Class (1.0) | **Reliability**: Percentage of predicted Heart Disease cases that were actually correct. |
            | **F1-Score** | Positive Class (1.0) | Harmonic mean of Precision and Recall. |
            | **Accuracy** | Global | Total percentage of correct predictions across both classes. |
            | **AUC** | Global | The model's ability to distinguish between classes across thresholds. |
            | **MCC** | Global | Correlation between observed and predicted classifications; best for imbalanced data. |
            """)
        
        # Create comparison table of all models' validation metrics
        test_results_df = pd.DataFrame({
            "Model": ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbor', 'Naive Bayes', 'Random Forest', 'XGBoost'],
            "Accuracy": [0.7559, 0.725, 0.7088, 0.7879, 0.7531, 0.7449],
            "AUC": [0.8458, 0.8189, 0.786, 0.8072, 0.8431, 0.8467],
            "Precision": [0.2499, 0.2268, 0.2093, 0.2462, 0.2462, 0.243],
            "Recall": [0.7957, 0.7974, 0.753, 0.6074, 0.7869, 0.8083],
            "F1": [0.3804, 0.3532, 0.3276, 0.3504, 0.3751, 0.3737],
            "MCC": [0.349, 0.3195, 0.2822, 0.2863, 0.3416, 0.3442]
        })
        st.table(test_results_df)

    st.markdown("---")
    
    # --- Deep Dive Section ---
    # Allow users to select and analyze a specific model in detail
    st.subheader("Model Deep Dive")
    st.write("Select a model to view detailed evaluation metrics and its Confusion Matrix.")
    
    # Dropdown to select model for detailed analysis
    model_choice = st.selectbox("Select Model for Analysis", list(detailed_metrics.keys()))
    
    # Retrieve detailed metrics for selected model from loaded dictionary
    data = detailed_metrics[model_choice]
    report = data['report']
    cm = np.array(data['matrix']) # Convert confusion matrix back to numpy for plotting

    # Create two columns for classification report and confusion matrix
    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Display classification report for selected model
        st.write(f"### Classification Report")
        
        # Convert classification report to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()

        # Extract and display overall accuracy metric
        if 'accuracy' in report_df.index:
            accuracy_value = report_df.loc['accuracy', 'precision']
            report_df = report_df.drop('accuracy')
            st.metric("Overall Model Accuracy", f"{accuracy_value:.2%}")

        # Define styling function to highlight the disease class row
        def highlight_target_class(x):
            # Style disease class (1.0) with blue background
            style = 'background-color: #1E3A8A; color: white; font-weight: bold;'
            return [style if x.name == '1.0' else '' for _ in x]

        # Apply styling and display report dataframe
        styled_report = report_df.style.format(precision=3).apply(highlight_target_class, axis=1)
        st.dataframe(styled_report)

    with col_b:
        # Display confusion matrix visualization
        st.write(f"### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease', 'Disease'], 
                    yticklabels=['No Disease', 'Disease'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

with tab2:
    # Tab 2: Live model testing and inference
    st.header("Model Evaluation Engine")
    
    # Section 1: Download test dataset
    st.subheader("1. Download Test Dataset")
    # Provide option to download raw test set for local testing
    try:
        with open(config['paths']['input_data']['test_data'], "rb") as file:
            st.download_button("Download Raw Test CSV", file, "heart_disease_raw_test_set.csv", "text/csv")
    except FileNotFoundError:
        st.warning("Raw test set CSV not found for download.")

    st.markdown("---")
    
    # Section 2: Upload CSV and run predictions
    st.subheader("2. Upload & Predict")
    uploaded_file = st.file_uploader("Upload the Test CSV file", type="csv")
    
    if uploaded_file:
        # Load uploaded CSV file
        df_raw = pd.read_csv(uploaded_file)
        # Select which model to use for predictions
        selected_name = st.selectbox("Select Model", list(models.keys()))
        selected_model = models[selected_name]
        
        # Button to trigger inference
        if st.button("Run Live Inference"):
            # Step 1: Feature Selection - Extract only the 16 features used in training
            X_input = df_raw[SELECTED_FEATURES]
            
            # Step 2: Scale features using trained scaler
            X_scaled = scaler.transform(X_input)
            
            # Step 3: Make predictions and get probability scores
            y_pred = selected_model.predict(X_scaled)
            y_probs = selected_model.predict_proba(X_scaled)[:, 1]
            
            # Step 4: Display success message
            st.success(f"Processed using {selected_name}")
            
            # Step 5: If ground truth target exists in uploaded file, calculate and display metrics
            target_col = 'HeartDiseaseorAttack'
            if target_col in df_raw.columns:
                y_true = df_raw[target_col]
                
                # Calculate all evaluation metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                auc = roc_auc_score(y_true, y_probs)
                
                # Display metrics in two rows of 3 columns each
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy:.2%}")
                m2.metric("Precision", f"{precision:.2%}")
                m3.metric("Recall", f"{recall:.2%}")
                
                m4, m5, m6 = st.columns(3)
                m4.metric("F1 Score", f"{f1:.2%}")
                m5.metric("MCC", f"{mcc:.3f}")
                m6.metric("AUC", f"{auc:.2%}")

                # Create two columns for detailed evaluation
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Display classification report
                    st.markdown("**Classification Report**")
                    report = classification_report(y_true, y_pred, output_dict=True)
                    if "accuracy" in report:
                        del report["accuracy"]
                    st.table(report)
                
                with col2:
                    # Display confusion matrix heatmap
                    st.markdown("**Confusion Matrix**")
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                
            else:
                # Show info message if no target column found
                st.info("No ground truth target column found in uploaded data for metric calculation.")
            
            # Display detailed prediction logs
            st.subheader("Detailed Prediction Logs (Top 20 Records)")
            st.write("This table shows the input health data alongside the model's calculated risk probability, final binary prediction along with the True label (HeartDiseaseorAttack).")
            # Add prediction and probability columns to results dataframe
            df_results = df_raw.copy()
            df_results['Prediction'] = y_pred
            df_results['Probability'] = y_probs
            st.dataframe(df_results.head(20))

            # --- Download Prediction Results ---
            # Section to export full results
            st.markdown("### Export Results")
            st.write("Click the button below to download the full set of predictions as a CSV file.")

            # Convert dataframe to CSV format for download
            csv_data = df_results.to_csv(index=False).encode('utf-8')

            # Download button for full prediction results
            st.download_button(
                label="Download Full Prediction Results",
                data=csv_data,
                file_name=f"heart_disease_predictions_{selected_name}.csv",
                mime="text/csv",
                help="Download a CSV file containing all original features plus the calculated Risk_Prediction and Probability columns."
            )