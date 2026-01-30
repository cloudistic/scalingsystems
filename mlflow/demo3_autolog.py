"""
Demo 3: MLflow Autologging
This script demonstrates MLflow's autologging feature for minimal code.
"""
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Enable autologging - this is the magic!
mlflow.sklearn.autolog()

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("autolog-demo")

print("=" * 60)
print("MLflow Demo 3: Autologging")
print("=" * 60)

print("\n1. Autologging enabled with one line:")
print("   mlflow.sklearn.autolog()")
print("   This will automatically log:")
print("   • All model parameters")
print("   • Training metrics")
print("   • Model artifacts")
print("   • Feature importance (when applicable)")

# Load data
print("\n2. Loading Iris dataset...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n3. Training model (autologging in action)...")
with mlflow.start_run(run_name="Random Forest - Autolog"):
    # Just train - MLflow logs everything automatically!
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    run_id = mlflow.active_run().info.run_id
    
    print(f"   ✅ Model trained successfully!")
    print(f"   Training Score: {train_score:.4f}")
    print(f"   Test Score: {test_score:.4f}")
    print(f"   Run ID: {run_id}")

print("\n4. What was logged automatically:")
print("   ✓ Model type and all parameters (n_estimators, max_depth, etc.)")
print("   ✓ Training and test scores")
print("   ✓ Model artifact (can be loaded later)")
print("   ✓ Feature importances")
print("   ✓ Training dataset signature")

print("\n5. View in MLflow UI:")
print("   • Open http://localhost:5000")
print("   • Navigate to 'autolog-demo' experiment")
print("   • Click on the run")
print("   • Notice all the logged information - with minimal code!")

print("\n💡 Key Takeaway:")
print("   Autologging captures MORE information with LESS code.")
print("   Perfect for production workflows!")
print("=" * 60)
