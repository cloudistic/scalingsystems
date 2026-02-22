"""
Demo 2: Model Comparison with MLflow
This script demonstrates how to compare multiple models using MLflow.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model-comparison")

print("=" * 60)
print("MLflow Demo 2: Model Comparison")
print("=" * 60)

# Load data
print("\n1. Loading Iris dataset...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

print(f"\n2. Training and comparing {len(models)} models...")
print("-" * 60)

results = []

# Train and log each model
for model_name, model in models.items():
    print(f"\n   Training: {model_name}")
    
    with mlflow.start_run(run_name=model_name):
        # Log model type
        mlflow.log_param("model_type", model_name)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        results.append({
            'name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1-Score:  {f1:.4f}")

print("\n" + "-" * 60)
print("\n3. Results Summary:")
print("-" * 60)
print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")
print("-" * 60)

# Find best model
best_model = max(results, key=lambda x: x['accuracy'])
print(f"\n✅ Best Model: {best_model['name']} (Accuracy: {best_model['accuracy']:.4f})")

print("\n4. View and compare in MLflow UI:")
print("   • Open http://localhost:5000")
print("   • Navigate to 'model-comparison' experiment")
print("   • Select all runs (checkboxes)")
print("   • Click 'Compare' button")
print("   • View parallel coordinates plot and scatter plots")
print("=" * 60)
