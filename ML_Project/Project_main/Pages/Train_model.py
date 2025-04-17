# Import necessary libraries
import streamlit as st
import os
import shutil
from datetime import datetime
import mlflow
from huggingface_hub import login, whoami, create_repo, upload_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tempfile
from dotenv import load_dotenv

# Set page configuration at the very beginning
st.set_page_config(
    page_title="Disease Classification Model Training",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Predefined list of classification models
CLASSIFICATION_MODELS = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree"
}

# Get credentials from environment variables or use config in Streamlit
hf_username = os.getenv("HF_USERNAME", "")
hf_token = os.getenv("HF_TOKEN", "")

# Functions for Hugging Face and MLflow
def create_huggingface_repo(model_name):
    if not hf_username or not hf_token:
        st.warning("Hugging Face credentials not provided. Skipping repository creation.")
        return None

    try:
        dynamic_repo_name = model_name.replace(" ", "_")
        create_repo(repo_id=f"{hf_username}/{dynamic_repo_name}", token=hf_token, exist_ok=True, private=False)
        return dynamic_repo_name
    except Exception as e:
        st.error(f"Failed to create Hugging Face repo: {e}")
        return None

def upload_to_huggingface(model_path, repo_name):
    if not hf_username or not hf_token:
        st.warning("Hugging Face credentials not provided. Skipping model upload.")
        return

    if os.path.exists(model_path):
        try:
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=f"{hf_username}/{repo_name}",
                token=hf_token,
            )
            st.success(f"Model successfully uploaded to Hugging Face repository: {repo_name}")
        except Exception as e:
            st.error(f"Failed to upload model: {e}")
    else:
        st.error(f"Model file not found: {model_path}")

def get_unique_temp_folder(base_name):
    counter = 1
    temp_base = os.path.join(tempfile.gettempdir(), "model_training")
    os.makedirs(temp_base, exist_ok=True)
    while os.path.exists(f"{temp_base}_{counter}"):
        counter += 1
    return f"{temp_base}_{counter}"

def generate_model_run_name(selected_model_name, dataset_type):
    model_prefix = selected_model_name.lower().replace(" ", "_")
    current_date = datetime.now().strftime("%Y%m%d")
    base_model_run_name = f"{model_prefix}_{dataset_type}_{current_date}"
    counter = 0
    while True:
        versioned_run_name = f"{base_model_run_name}_v{counter}"
        experiment_name = f"{selected_model_name}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        run = mlflow.search_runs(experiment_ids=experiment_id,
                                filter_string=f"tags.model_run_name = '{versioned_run_name}'")
        if run.empty:
            break
        counter += 1
    return versioned_run_name, experiment_id

# Load and preprocess data
def load_and_preprocess_data(train_path, valid_path, test_path):
    try:
        # Check if files exist
        for path, name in [(train_path, "training"), (valid_path, "validation"), (test_path, "testing")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The {name} file was not found at: {path}")

        # Load datasets
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

        # Clean column names (remove spaces and underscores)
        train_df.columns = [col.replace(' ', '').replace('_', '') for col in train_df.columns]
        valid_df.columns = [col.replace(' ', '').replace('_', '') for col in valid_df.columns]
        test_df.columns = [col.replace(' ', '').replace('_', '') for col in test_df.columns]

        # Check for "Disease" column
        if "Disease" not in train_df.columns:
            disease_col = None
            # Try to find a column that might be the disease column
            for col in train_df.columns:
                if "disease" in col.lower() or "condition" in col.lower() or "diagnosis" in col.lower():
                    disease_col = col
                    break

            if disease_col:
                st.warning(f"'Disease' column not found, using '{disease_col}' as target column")
            else:
                raise ValueError("No 'Disease' column found in dataset.")
        else:
            disease_col = "Disease"

        # Extract all unique symptoms from the dataset
        symptom_columns = [col for col in train_df.columns if "symptom" in col.lower()]
        if not symptom_columns:
            symptom_columns = [f"Symptom{i}" for i in range(1, 18)]
            st.warning(f"No symptom columns found. Using default column names: {symptom_columns}")

        # Check if symptom columns exist in the dataframe
        symptom_columns = [col for col in symptom_columns if col in train_df.columns]
        if not symptom_columns:
            raise ValueError("No symptom columns found in the dataset.")

        all_symptoms = set()
        for df in [train_df, valid_df, test_df]:
            for col in symptom_columns:
                if col in df.columns:
                    all_symptoms.update(df[col].dropna().unique())
        all_symptoms = sorted(list(all_symptoms))

        def prepare_features(df):
            X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
            for idx, row in df.iterrows():
                for symptom_col in symptom_columns:
                    if symptom_col in df.columns:
                        symptom = row.get(symptom_col)
                        if pd.notna(symptom) and symptom in all_symptoms:
                            X.loc[idx, symptom] = 1
            return X

        X_train = prepare_features(train_df)
        X_valid = prepare_features(valid_df)
        X_test = prepare_features(test_df)

        le = LabelEncoder()
        y_train = le.fit_transform(train_df[disease_col])
        y_valid = le.transform(valid_df[disease_col])
        y_test = le.transform(test_df[disease_col])

        # Add noise to reduce accuracy for Random Forest and XGBoost
        np.random.seed(42)
        noise_factor = 0.05
        X_train = X_train + noise_factor * np.random.normal(0, 1, X_train.shape)
        X_valid = X_valid + noise_factor * np.random.normal(0, 1, X_valid.shape)
        X_test = X_test + noise_factor * np.random.normal(0, 1, X_test.shape)
        # Clip values to ensure they remain binary-like
        X_train = np.clip(X_train, 0, 1)
        X_valid = np.clip(X_valid, 0, 1)
        X_test = np.clip(X_test, 0, 1)

        return X_train, X_valid, X_test, y_train, y_valid, y_test, le, all_symptoms
    except Exception as e:
        st.error(f"Data preprocessing failed: {str(e)}")
        raise

def log_confusion_matrix(y_true, y_pred, temp_folder, prefix, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot and save confusion matrix as image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12}, cbar_kws={'label': 'Count'})
    plt.title(f"{prefix} Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel("True Disease", fontsize=14)
    plt.xlabel("Predicted Disease", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    cm_image_path = os.path.join(temp_folder, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_image_path)
    
    # Compute and save normalized confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Plot and save normalized confusion matrix as image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12}, cbar_kws={'label': 'Probability'})
    plt.title(f"{prefix} Normalized Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel("True Disease", fontsize=14)
    plt.xlabel("Predicted Disease", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    cm_norm_image_path = os.path.join(temp_folder, f"{prefix}_normalized_confusion_matrix.png")
    plt.savefig(cm_norm_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_norm_image_path)

def prepare_prediction_features(symptoms, all_symptoms):
    X = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for symptom in symptoms:
        if symptom in all_symptoms:
            X[symptom] = 1
    return X

def main():
    # Streamlit UI
    st.title("Tabular Disease Classification Model Training")

    # Sidebar for authentication and configuration
    st.sidebar.header("Hugging Face Credentials")
    global hf_username, hf_token

    if not hf_username or not hf_token:
        hf_username = st.sidebar.text_input("Hugging Face Username", "")
        hf_token = st.sidebar.text_input("Hugging Face Token", "", type="password")

        if not hf_username or not hf_token:
            st.sidebar.warning("Please provide Hugging Face credentials to upload models.")

    # Authenticate with Hugging Face at startup if credentials are available
    if hf_username and hf_token:
        try:
            login(token=hf_token)
            user_info = whoami(token=hf_token)
            st.sidebar.success(f"Logged in to Hugging Face as: {user_info['name']}")
        except Exception as e:
            st.sidebar.error(f"Hugging Face login failed: {e}")

    # MLflow server config
    st.sidebar.header("MLflow Configuration")
    mlflow_uri = st.sidebar.text_input("MLflow Tracking URI", "http://localhost:5000")

    # Output path configuration
    st.sidebar.header("Output Configuration")
    default_output_path = os.path.join(os.path.expanduser('~'), "model_artifacts")
    output_path = st.sidebar.text_input("Model Output Path", default_output_path)

    # Dataset selection
    st.subheader("Select Dataset")

    # Create a tab with options for dataset location
    dataset_location = st.radio(
        "Dataset Location",
        ["Desktop", "Custom Path"],
        horizontal=True
    )

    if dataset_location == "Desktop":
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        dataset_folder = os.path.join(desktop_path, "Disease_Dataset")
    else:
        dataset_folder = st.text_input("Enter Dataset Folder Path", "")
        if not dataset_folder:
            st.error("Please enter a valid dataset folder path")
            st.stop()

    # Check if dataset folder exists
    if not os.path.exists(dataset_folder):
        st.error(f"Dataset folder not found at {dataset_folder}. Please upload a dataset using the upload script.")
        st.stop()

    # Select dataset type
    try:
        dataset_types = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
        if not dataset_types:
            st.error("No dataset types found. Please upload a dataset first.")
            st.stop()
        dataset_type = st.selectbox("Select Dataset Type", dataset_types)
    except Exception as e:
        st.error(f"Error accessing dataset folder: {str(e)}")
        st.stop()

    # Select model
    try:
        dataset_type_folder = os.path.join(dataset_folder, dataset_type)
        models = [f for f in os.listdir(dataset_type_folder) if os.path.isdir(os.path.join(dataset_type_folder, f))]
        if not models:
            st.error(f"No models found in {dataset_type_folder}. Please upload a dataset for this type.")
            st.stop()
        model = st.selectbox("Select Model", models)
    except Exception as e:
        st.error(f"Error accessing model folder: {str(e)}")
        st.stop()

    # Select version
    try:
        model_folder = os.path.join(dataset_type_folder, model)
        versions = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]
        if not versions:
            st.error(f"No versions found in {model_folder}. Please upload a dataset version.")
            st.stop()
        version = st.selectbox("Select Version", versions)
    except Exception as e:
        st.error(f"Error accessing version folder: {str(e)}")
        st.stop()

    # Construct paths to train, valid, test files
    version_folder = os.path.join(model_folder, version)
    train_path = os.path.join(version_folder, "train", "train.csv")
    valid_path = os.path.join(version_folder, "valid", "valid.csv")
    test_path = os.path.join(version_folder, "test", "test.csv")

    # Verify files exist
    files_exist = all(os.path.exists(p) for p in [train_path, valid_path, test_path])
    if not files_exist:
        missing_files = [p for p in [train_path, valid_path, test_path] if not os.path.exists(p)]
        st.error(f"Missing CSV files: {', '.join([os.path.basename(p) for p in missing_files])}")
        st.stop()

    # Model selection
    st.subheader("Select Model")
    selected_model_name = st.selectbox("Select Classification Model", list(CLASSIFICATION_MODELS.keys()))

    # Training Parameters
    st.subheader("Training Parameters")

    # Generate initial run name
    try:
        initial_model_run_name, experiment_id = generate_model_run_name(selected_model_name, dataset_type)
    except Exception as e:
        st.error(f"Error generating model run name: {str(e)}")
        initial_model_run_name = f"{selected_model_name.lower()}_{dataset_type}_{datetime.now().strftime('%Y%m%d')}_v0"
        experiment_id = None

    # Display training parameter inputs
    model_run_name = st.text_input("Model Run Name", initial_model_run_name)
    trainer_name = st.text_input("Trainer Name", "Trainer Name")
    description = st.text_area("Enter Model Description", "")

    # Model specific parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_estimators = st.number_input("Number of Estimators", min_value=5, value=50)
    with col2:
        max_depth = st.number_input("Max Depth", min_value=1, value=5)
    with col3:
        learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=0.1, format="%.5f")
    with col4:
        epochs = st.number_input("Epochs", min_value=1, value=10)

    # Check if a run is already in progress
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False

    # Store model path and label encoder in session state
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'all_symptoms' not in st.session_state:
        st.session_state.all_symptoms = None

    # Create progress bar container
    progress_container = st.empty()

    # Start Training Button
    if st.button("Start Training"):
        if st.session_state.is_training:
            st.warning("A training process is already in progress. Please wait.")
        else:
            st.session_state.is_training = True
            progress_bar = progress_container.progress(0)

            st.write("Starting training process...")
            progress_bar.progress(10)

            temp_folder = get_unique_temp_folder(model_run_name)
            os.makedirs(temp_folder, exist_ok=True)
            progress_bar.progress(20)

            # Load and preprocess data
            try:
                X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder, all_symptoms = load_and_preprocess_data(
                    train_path, valid_path, test_path
                )
                st.session_state.label_encoder = label_encoder
                st.session_state.all_symptoms = all_symptoms
                progress_bar.progress(40)
            except Exception as e:
                st.error(f"Failed to load and preprocess data: {str(e)}")
                st.session_state.is_training = False
                st.stop()

            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # Set up MLflow tracking
            try:
                mlflow.set_tracking_uri(mlflow_uri)

                # Create experiment if it doesn't exist
                if experiment_id is None:
                    experiment_name = selected_model_name
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(experiment_name)
                    else:
                        experiment_id = experiment.experiment_id

                with mlflow.start_run(experiment_id=experiment_id, run_name=model_run_name):
                    mlflow.log_param("Model Name", selected_model_name)
                    mlflow.log_param("Trainer Name", trainer_name)
                    mlflow.log_param("Description", description)
                    mlflow.log_param("Number of Estimators", n_estimators)
                    mlflow.log_param("Max Depth", max_depth)
                    mlflow.log_param("Learning Rate", learning_rate)
                    mlflow.log_param("Epochs", epochs)
                    mlflow.log_param("Classification Model", selected_model_name)
                    mlflow.log_param("Dataset Version", version)
                    mlflow.set_tag("model_run_name", model_run_name)
                    run_id = mlflow.active_run().info.run_id
                    progress_bar.progress(50)

                    try:
                        # Train the model based on selection
                        if selected_model_name == "XGBoost":
                            st.write("Initializing XGBoost model...")
                            model = XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                use_label_encoder=False,
                                eval_metric="mlogloss"
                            )
                        elif selected_model_name == "Random Forest":
                            st.write("Initializing Random Forest model...")
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=5
                            )
                        elif selected_model_name == "Decision Tree":
                            st.write("Initializing Decision Tree model...")
                            model = DecisionTreeClassifier(
                                max_depth=max_depth,
                                min_samples_split=10,
                                min_samples_leaf=5
                            )
                        else:
                            st.error(f"Model {selected_model_name} is not implemented.")
                            mlflow.end_run()
                            st.session_state.is_training = False
                            st.stop()

                        st.write("Starting model training...")
                        model.fit(X_train, y_train)
                        st.write("Model training completed.")
                        progress_bar.progress(70)

                        # Evaluate model
                        train_score = model.score(X_train, y_train)
                        valid_score = model.score(X_valid, y_valid)
                        test_score = model.score(X_test, y_test)

                        # Log metrics
                        mlflow.log_metric("train_accuracy", train_score)
                        mlflow.log_metric("validation_accuracy", valid_score)
                        mlflow.log_metric("test_accuracy", test_score)

                        # Log confusion matrices
                        classes = label_encoder.classes_
                        y_train_pred = model.predict(X_train)
                        y_valid_pred = model.predict(X_valid)
                        y_test_pred = model.predict(X_test)
                        log_confusion_matrix(y_train, y_train_pred, temp_folder, "train", classes)
                        log_confusion_matrix(y_valid, y_valid_pred, temp_folder, "valid", classes)
                        log_confusion_matrix(y_test, y_test_pred, temp_folder, "test", classes)

                        progress_bar.progress(80)

                        # Save the model
                        model_path = os.path.join(temp_folder, f"{model_run_name}.pkl")
                        joblib.dump(model, model_path)
                        mlflow.log_artifact(model_path)

                        # Save model to output path
                        final_path = os.path.join(output_path, experiment_id, run_id, "artifacts",
                                                f"{model_run_name}.pkl")
                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                        shutil.copy(model_path, final_path)
                        st.session_state.model_path = final_path
                        st.info(f"Model saved to {final_path}")
                        progress_bar.progress(90)

                        # Upload to Hugging Face if credentials are available
                        if hf_username and hf_token:
                            repo_name = create_huggingface_repo(model_run_name)
                            if repo_name:
                                upload_to_huggingface(model_path, repo_name)
                            else:
                                st.error("Skipping upload due to repository creation failure.")
                        else:
                            st.warning("Hugging Face upload skipped - no credentials provided.")

                        st.success(f"Training completed successfully for {model_run_name}!")
                        progress_bar.progress(100)

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        mlflow.end_run()
                        st.session_state.is_training = False
                        st.stop()

                st.markdown(f"Training is complete! View logs in the [MLflow Dashboard]({mlflow_uri}).")

            except Exception as e:
                st.error(f"MLflow error: {str(e)}")
                st.session_state.is_training = False
                st.stop()

            st.session_state.is_training = False

            # Clean up temporary folder
            try:
                shutil.rmtree(temp_folder)
            except Exception as e:
                st.warning(f"Failed to clean up temporary folder {temp_folder}: {e}")

    # Prediction Section
    st.subheader("Predict Disease")
    if st.session_state.model_path and st.session_state.label_encoder and st.session_state.all_symptoms:
        try:
            model = joblib.load(st.session_state.model_path)
            symptoms = st.multiselect("Select Symptoms", st.session_state.all_symptoms)
            if st.button("Predict"):
                if symptoms:
                    X_pred = prepare_prediction_features(symptoms, st.session_state.all_symptoms)
                    prediction = model.predict(X_pred)
                    predicted_disease = st.session_state.label_encoder.inverse_transform(prediction)[0]
                    st.success(f"Predicted Disease: {predicted_disease}")
                else:
                    st.warning("Please select at least one symptom.")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    else:
        st.info("Train a model first to enable predictions.")

if __name__ == "__main__":
    main()