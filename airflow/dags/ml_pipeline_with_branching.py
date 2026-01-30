"""
ML Pipeline with Conditional Branching
Demonstrates how to use branching logic in Airflow based on data quality checks.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
import random

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    dag_id='ml_pipeline_with_branching',
    default_args=default_args,
    description='ML pipeline demonstrating conditional branching based on data quality',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'advanced', 'branching'],
)

# Task Functions

def check_data_quality(**context):
    """
    Check data quality and decide which pipeline path to take.
    This is a BranchPythonOperator that returns the task_id to execute next.
    """
    # Simulate data quality check (in production, this would analyze actual data)
    quality_score = random.uniform(0, 1)
    
    print("=" * 60)
    print("🔍 DATA QUALITY CHECK")
    print("=" * 60)
    print(f"Quality Score: {quality_score:.2f}")
    
    # Store quality score in XCom for other tasks to access
    context['task_instance'].xcom_push(key='quality_score', value=quality_score)
    
    # Decision logic - return the task_id that should execute next
    if quality_score > 0.8:
        print("✅ HIGH QUALITY - Using full pipeline")
        print("=" * 60)
        return 'high_quality_pipeline'
    elif quality_score > 0.5:
        print("⚠️  MEDIUM QUALITY - Using simplified pipeline")
        print("=" * 60)
        return 'medium_quality_pipeline'
    else:
        print("❌ LOW QUALITY - Sending alert")
        print("=" * 60)
        return 'data_quality_alert'

def high_quality_pipeline(**context):
    """Execute full feature engineering for high-quality data"""
    # Retrieve quality score from XCom
    ti = context['task_instance']
    quality_score = ti.xcom_pull(key='quality_score', task_ids='check_data_quality')
    
    print("=" * 60)
    print("🚀 HIGH QUALITY PIPELINE")
    print("=" * 60)
    print(f"Data Quality Score: {quality_score:.2f}")
    print("\nExecuting full pipeline:")
    print("  1. Advanced feature engineering")
    print("  2. Complex transformations")
    print("  3. Multiple feature interactions")
    print("  4. Deep learning model training")
    print("=" * 60)
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    print("✅ Full pipeline completed successfully!")

def medium_quality_pipeline(**context):
    """Execute simplified feature engineering for medium-quality data"""
    # Retrieve quality score from XCom
    ti = context['task_instance']
    quality_score = ti.xcom_pull(key='quality_score', task_ids='check_data_quality')
    
    print("=" * 60)
    print("⚙️  MEDIUM QUALITY PIPELINE")
    print("=" * 60)
    print(f"Data Quality Score: {quality_score:.2f}")
    print("\nExecuting simplified pipeline:")
    print("  1. Basic feature engineering")
    print("  2. Standard transformations")
    print("  3. Simple model training")
    print("=" * 60)
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    print("✅ Simplified pipeline completed successfully!")

def send_data_quality_alert(**context):
    """Send alert for low-quality data"""
    # Retrieve quality score from XCom
    ti = context['task_instance']
    quality_score = ti.xcom_pull(key='quality_score', task_ids='check_data_quality')
    
    print("=" * 60)
    print("⚠️  DATA QUALITY ALERT")
    print("=" * 60)
    print(f"Data Quality Score: {quality_score:.2f}")
    print(f"Threshold: 0.50")
    print(f"\n❌ ALERT: Data quality is below acceptable threshold!")
    print("\nAction Required:")
    print("  • Review data source")
    print("  • Check data pipeline")
    print("  • Investigate data quality issues")
    print("  • Manual intervention needed")
    print("\n📧 Alert sent to: ml-team@example.com")
    print("=" * 60)

def final_summary(**context):
    """Provide a summary of the pipeline execution"""
    ti = context['task_instance']
    quality_score = ti.xcom_pull(key='quality_score', task_ids='check_data_quality')
    
    # Determine which path was taken
    if quality_score > 0.8:
        path = "High Quality Pipeline"
    elif quality_score > 0.5:
        path = "Medium Quality Pipeline"
    else:
        path = "Data Quality Alert"
    
    print("\n" + "=" * 60)
    print("📋 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Quality Score: {quality_score:.2f}")
    print(f"Pipeline Path Taken: {path}")
    print("=" * 60 + "\n")

# Define tasks

# Task 1: Initial data ingestion (always runs)
task_ingest = BashOperator(
    task_id='ingest_data',
    bash_command='echo "📥 Ingesting data from source..."',
    dag=dag,
)

# Task 2: Check data quality and branch (BranchPythonOperator)
task_check = BranchPythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

# Branch 1: High quality data pipeline
task_high_quality = PythonOperator(
    task_id='high_quality_pipeline',
    python_callable=high_quality_pipeline,
    dag=dag,
)

# Branch 2: Medium quality data pipeline
task_medium_quality = PythonOperator(
    task_id='medium_quality_pipeline',
    python_callable=medium_quality_pipeline,
    dag=dag,
)

# Branch 3: Data quality alert
task_alert = PythonOperator(
    task_id='data_quality_alert',
    python_callable=send_data_quality_alert,
    dag=dag,
)

# Task 7: Final summary (runs after any branch completes)
task_final = PythonOperator(
    task_id='final_summary',
    python_callable=final_summary,
    trigger_rule='none_failed',  # Run if no upstream tasks failed
    dag=dag,
)

# Define the branching workflow
#
# Workflow visualization:
#
#                           ┌─→ high_quality_pipeline ─┐
#                           │                          │
# ingest_data → check_quality ─→ medium_quality_pipeline ─→ final_summary
#                           │                          │
#                           └─→ data_quality_alert ────┘
#

task_ingest >> task_check
task_check >> [task_high_quality, task_medium_quality, task_alert]
task_high_quality >> task_final
task_medium_quality >> task_final
task_alert >> task_final
