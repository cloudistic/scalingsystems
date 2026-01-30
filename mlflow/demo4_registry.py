"""
Demo 4: MLflow Model Registry
This script demonstrates how to register and manage models in MLflow's Model Registry.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model-registry-demo")

print("=" * 60)
print("MLflow Demo 4: Model Registry")
print("=" * 60)

# Load and prepare data
print("\n1. Preparing data...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a good model
print("\n2. Training a production-ready model...")
with mlflow.start_run(run_name="Production Candidate") as run:
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)
    
    # Register the model in Model Registry
    print("\n3. Registering model in Model Registry...")
    model_uri = f"runs:/{run.info.run_id}/model"
    registered_model = mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="iris-production-model"
    )
    
    print(f"   ✅ Model registered successfully!")
    print(f"   Model Name: iris-production-model")
    print(f"   Version: {registered_model.registered_model_version}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    run_id = run.info.run_id

# Manage model versions
print("\n4. Managing model lifecycle...")
client = MlflowClient()

# Transition model to staging
print("   • Transitioning model to 'Staging'...")
client.transition_model_version_stage(
    name="iris-production-model",
    version=1,
    stage="Staging"
)
print("   ✅ Model is now in Staging")

# Add description
print("   • Adding model description...")
client.update_registered_model(
    name="iris-production-model",
    description="Iris classification model using Random Forest. "
                "Trained on iris dataset with 100 estimators and max_depth=5."
)

# Add version description
client.update_model_version(
    name="iris-production-model",
    version=1,
    description=f"Baseline model. Accuracy: {accuracy:.4f}"
)
print("   ✅ Description added")

# Demonstrate loading from registry
print("\n5. Loading model from registry...")
staging_model_uri = "models:/iris-production-model/Staging"
loaded_model = mlflow.pyfunc.load_model(staging_model_uri)

# Test prediction
import numpy as np
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example iris measurement
prediction = loaded_model.predict(sample_data)
print(f"   ✅ Model loaded from registry")
print(f"   Sample prediction: {prediction}")

print("\n6. Model Registry Benefits:")
print("   ✓ Version control for models")
print("   ✓ Stage management (Staging → Production)")
print("   ✓ Model lineage tracking")
print("   ✓ Easy model deployment")
print("   ✓ Approval workflows")

print("\n7. View in MLflow UI:")
print("   • Open http://localhost:5000")
print("   • Click on 'Models' tab")
print("   • Find 'iris-production-model'")
print("   • Explore versions, stages, and metadata")

print("\n8. Next steps - Transition to Production:")
print("   • After validation in staging")
print("   • Click 'Stage' dropdown → 'Transition to Production'")
print("   • Production services can load: models:/iris-production-model/Production")

print("=" * 60)
