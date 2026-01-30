# MLOps Scaling Systems - 2-Hour Lecture Series

This repository contains hands-on demonstrations for key MLOps tools, designed for a 2-hour lecture targeted at analytical industry professionals. Each demo is simple, well-documented, and demonstrates core capabilities of essential MLOps tools.

## 📚 What's Inside

This repository demonstrates three critical MLOps tools:

1. **DVC (Data Version Control)** - Version control for data and ML models
2. **MLflow** - Experiment tracking, model registry, and deployment
3. **Apache Airflow** - Workflow orchestration for ML pipelines

## 🎯 Learning Objectives

By the end of this lecture series, you will understand how to:
- Version control your data and models alongside your code
- Track ML experiments systematically
- Register and manage ML models
- Orchestrate complex ML workflows
- Build reproducible ML pipelines

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic knowledge of Python and Machine Learning
- Git installed on your system

### Installation

1. Clone this repository:
```bash
git clone https://github.com/cloudistic/scalingsystems.git
cd scalingsystems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Repository Structure

```
scalingsystems/
├── dvc/              # DVC demonstrations
├── mlflow/           # MLflow demonstrations  
├── airflow/          # Airflow DAG examples
├── h20ai/            # H2O AutoML examples
├── tpot/             # TPOT AutoML examples
└── README.md         # This file
```

## 📖 Lecture Modules

### Module 1: DVC - Data Version Control (30 minutes)
**Location:** `dvc/`

Learn how to:
- Track large datasets with DVC
- Version ML models
- Create reproducible ML pipelines
- Share data efficiently across teams

[👉 Start DVC Tutorial](dvc/README.md)

### Module 2: MLflow - Experiment Tracking (45 minutes)
**Location:** `mlflow/`

Learn how to:
- Track experiments and parameters
- Log metrics and artifacts
- Register and version models
- Compare multiple runs

[👉 Start MLflow Tutorial](mlflow/README.md)

### Module 3: Apache Airflow - Workflow Orchestration (45 minutes)
**Location:** `airflow/`

Learn how to:
- Create ML pipeline DAGs
- Schedule training workflows
- Monitor pipeline execution
- Handle dependencies and retries

[👉 Start Airflow Tutorial](airflow/README.md)

## 🎓 Recommended Learning Path

For a 2-hour lecture session, follow this sequence:

1. **Introduction (10 min)** - MLOps overview and challenges
2. **DVC Demo (30 min)** - Data versioning and pipeline management
3. **MLflow Demo (45 min)** - Experiment tracking and model registry
4. **Airflow Demo (30 min)** - ML workflow orchestration
5. **Q&A and Wrap-up (5 min)**

## 💡 Best Practices Demonstrated

- **Version Everything:** Code, data, models, and configurations
- **Track Experiments:** Never lose track of what you tried
- **Automate Pipelines:** Reduce manual errors and save time
- **Reproducibility:** Ensure results can be recreated
- **Collaboration:** Share work effectively with teams

## 🔗 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)

## 📝 License

This repository is for educational purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve these demos.

---
**Note:** These demos are simplified for teaching purposes. Production implementations may require additional considerations for security, scalability, and robustness.
