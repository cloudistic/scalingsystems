"""
Complete ML Training Pipeline DAG
This DAG demonstrates an end-to-end machine learning workflow using Airflow.
Includes: data extraction, validation, preprocessing, training, evaluation, and deployment.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import pickle
import json
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Default arguments for all tasks
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline with Iris dataset',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'production', 'iris'],
)

# Configuration
DATA_DIR = '/tmp/ml_pipeline'
MIN_ACCURACY_THRESHOLD = 0.85

# Task Functions

def create_directories():
    """Task 1: Create necessary directories for the pipeline"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"✅ Created directory: {DATA_DIR}")

def extract_data():
    """Task 2: Extract data from source (simulated with iris dataset)"""
    print("📥 Extracting data from source...")
    
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    df['target_name'] = df['target'].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )
    
    # Save raw data
    output_path = f"{DATA_DIR}/raw_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Data extracted successfully")
    print(f"   • Output: {output_path}")
    print(f"   • Shape: {df.shape}")
    print(f"   • Columns: {list(df.columns)}")
    
    return output_path

def validate_data():
    """Task 3: Validate data quality and integrity"""
    print("🔍 Validating data quality...")
    
    df = pd.read_csv(f"{DATA_DIR}/raw_data.csv")
    
    # Check 1: Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"❌ Data contains {missing} missing values!")
    print(f"   ✓ No missing values")
    
    # Check 2: Sufficient samples
    if df.shape[0] < 100:
        raise ValueError(f"❌ Insufficient data: only {df.shape[0]} samples!")
    print(f"   ✓ Sufficient samples: {df.shape[0]}")
    
    # Check 3: Feature count
    expected_features = 5  # 4 features + 1 target
    if df.shape[1] < expected_features:
        raise ValueError(f"❌ Expected {expected_features} columns, got {df.shape[1]}")
    print(f"   ✓ Correct feature count: {df.shape[1]}")
    
    # Check 4: Target distribution
    target_dist = df['target'].value_counts()
    print(f"   ✓ Target distribution:")
    for target, count in target_dist.items():
        print(f"      - Class {target}: {count} samples")
    
    print("✅ Data validation passed!")

def preprocess_data():
    """Task 4: Preprocess data and create train/test split"""
    print("⚙️  Preprocessing data...")
    
    df = pd.read_csv(f"{DATA_DIR}/raw_data.csv")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"   • Features: {list(X.columns)}")
    print(f"   • Target: unique classes = {y.nunique()}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)
    
    print(f"✅ Data preprocessing completed")
    print(f"   • Train samples: {X_train.shape[0]}")
    print(f"   • Test samples: {X_test.shape[0]}")
    print(f"   • Features: {X_train.shape[1]}")

def train_model():
    """Task 5: Train the machine learning model"""
    print("🎓 Training model...")
    
    # Load training data
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    
    # Model hyperparameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 2,
        'random_state': 42
    }
    
    print(f"   • Model: RandomForestClassifier")
    print(f"   • Parameters: {model_params}")
    
    # Train model
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = f"{DATA_DIR}/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'parameters': model_params,
        'training_samples': len(X_train),
        'features': list(X_train.columns),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{DATA_DIR}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model trained successfully")
    print(f"   • Model saved: {model_path}")
    print(f"   • Training samples: {len(X_train)}")

def evaluate_model():
    """Task 6: Evaluate model performance on test data"""
    print("📊 Evaluating model...")
    
    # Load test data
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()
    
    # Load model
    with open(f"{DATA_DIR}/model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   • Test Accuracy: {accuracy:.4f}")
    print(f"   • Precision: {report['weighted avg']['precision']:.4f}")
    print(f"   • Recall: {report['weighted avg']['recall']:.4f}")
    print(f"   • F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1_score': float(report['weighted avg']['f1-score']),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{DATA_DIR}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Model evaluation completed")
    print(f"   • Metrics saved: {DATA_DIR}/metrics.json")
    
    # Quality gate: Check if accuracy meets threshold
    if accuracy < MIN_ACCURACY_THRESHOLD:
        raise ValueError(
            f"❌ Model accuracy {accuracy:.4f} is below threshold {MIN_ACCURACY_THRESHOLD}!"
        )
    
    print(f"   ✓ Quality gate passed (accuracy > {MIN_ACCURACY_THRESHOLD})")

def deploy_model():
    """Task 7: Deploy model to production (simulated)"""
    print("🚀 Deploying model...")
    
    import shutil
    
    # In production, this would deploy to a model serving platform
    # For this demo, we simulate by copying to a "production" location
    
    source_model = f"{DATA_DIR}/model.pkl"
    source_metadata = f"{DATA_DIR}/model_metadata.json"
    
    prod_model = f"{DATA_DIR}/production_model.pkl"
    prod_metadata = f"{DATA_DIR}/production_metadata.json"
    
    shutil.copy(source_model, prod_model)
    shutil.copy(source_metadata, prod_metadata)
    
    # Create deployment record
    deployment_info = {
        'deployed_at': datetime.now().isoformat(),
        'model_path': prod_model,
        'metadata_path': prod_metadata,
        'status': 'deployed'
    }
    
    with open(f"{DATA_DIR}/deployment_info.json", 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"✅ Model deployed successfully")
    print(f"   • Production model: {prod_model}")
    print(f"   • Deployment time: {deployment_info['deployed_at']}")

def send_notification():
    """Task 8: Send pipeline completion notification"""
    print("📧 Sending notification...")
    
    # Load metrics for the notification
    with open(f"{DATA_DIR}/metrics.json", 'r') as f:
        metrics = json.load(f)
    
    # In production, this would send an email, Slack message, etc.
    # For demo, we print a summary
    
    print("\n" + "=" * 60)
    print("🎉 ML PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 60)
    print(f"Pipeline: ml_training_pipeline")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Model Performance:")
    print(f"   • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   • Precision: {metrics['precision']:.4f}")
    print(f"   • Recall:    {metrics['recall']:.4f}")
    print(f"   • F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\n✅ Model has been deployed to production")
    print("=" * 60 + "\n")

# Define tasks
task_create_dirs = PythonOperator(
    task_id='create_directories',
    python_callable=create_directories,
    dag=dag,
)

task_extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

task_deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# Define the workflow pipeline
# Sequential execution: each task depends on the previous one
task_create_dirs >> task_extract >> task_validate >> task_preprocess
task_preprocess >> task_train >> task_evaluate >> task_deploy >> task_notify

# Workflow visualization:
#
# create_directories → extract_data → validate_data → preprocess_data
#                                                           ↓
#                                                      train_model
#                                                           ↓
#                                                    evaluate_model
#                                                           ↓
#                                                     deploy_model
#                                                           ↓
#                                                  send_notification
