"""
Simple Hello World DAG for Airflow Tutorial
This is a beginner-friendly example demonstrating basic Airflow concepts.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments applied to all tasks
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    dag_id='hello_world',
    default_args=default_args,
    description='A simple hello world DAG for learning Airflow basics',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't backfill past dates
    tags=['tutorial', 'beginner'],
)

# Python functions that will be executed as tasks
def print_hello():
    """Print a hello message"""
    print("=" * 50)
    print("Hello from Apache Airflow!")
    print("This is a Python task executing successfully")
    print("=" * 50)
    return "Hello task completed"

def print_date():
    """Print current date and time"""
    current_time = datetime.now()
    print("=" * 50)
    print(f"Current Date and Time: {current_time}")
    print(f"Date: {current_time.strftime('%Y-%m-%d')}")
    print(f"Time: {current_time.strftime('%H:%M:%S')}")
    print("=" * 50)
    return "Date task completed"

def print_context(**context):
    """Print task context information"""
    print("=" * 50)
    print("Task Context Information:")
    print(f"Execution Date: {context['execution_date']}")
    print(f"DAG ID: {context['dag'].dag_id}")
    print(f"Task ID: {context['task'].task_id}")
    print("=" * 50)
    return "Context task completed"

# Task 1: Say Hello using Python
task_hello = PythonOperator(
    task_id='say_hello',
    python_callable=print_hello,
    dag=dag,
)

# Task 2: Print current date using Python
task_date = PythonOperator(
    task_id='print_current_date',
    python_callable=print_date,
    dag=dag,
)

# Task 3: Print context information
task_context = PythonOperator(
    task_id='print_context',
    python_callable=print_context,
    dag=dag,
)

# Task 4: Execute bash command
task_bash = BashOperator(
    task_id='bash_hello',
    bash_command='echo "Hello from Bash! Pipeline execution successful."',
    dag=dag,
)

# Task 5: Final task
task_final = BashOperator(
    task_id='final_task',
    bash_command='echo "All tasks completed successfully! ✅"',
    dag=dag,
)

# Define task dependencies (execution order)
# This creates the workflow: hello → date → context → bash → final
task_hello >> task_date >> task_context >> task_bash >> task_final

# Alternative syntax for parallel execution:
# task_hello >> [task_date, task_context] >> task_bash >> task_final
# This would run task_date and task_context in parallel after task_hello
