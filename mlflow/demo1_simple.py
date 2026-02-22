"""
Demo 1: Simple MLflow Experiment Tracking
This script demonstrates the basics of MLflow experiment tracking.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("lecture-demo")

print("=" * 60)
print("MLflow Demo 1: Simple Experiment Tracking")
print("=" * 60)

# Load data
print("\n1. Loading Iris dataset...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Train and log with MLflow
print("\n2. Training model and logging to MLflow...")
with mlflow.start_run(run_name="Logistic Regression - Demo 1"):
    # Log parameters
    max_iter = 100
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("solver", "lbfgs")
    
    # Train model
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    
    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    run_id = mlflow.active_run().info.run_id
    
    print(f"   ✅ Model trained successfully!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Run ID: {run_id}")

print("\n3. View results:")
print("   Open http://localhost:5000 in your browser")
print("   Navigate to 'lecture-demo' experiment")
print("   Click on the run to see parameters, metrics, and model")
print("=" * 60)
