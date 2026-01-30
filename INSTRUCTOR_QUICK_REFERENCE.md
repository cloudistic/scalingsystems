# MLOps Lecture - Quick Reference Card for Instructors

## 🎯 Before the Lecture

### Day Before
- [ ] Test all demos on your machine
- [ ] Start MLflow server: `mlflow server --host 0.0.0.0 --port 5000`
- [ ] Initialize Airflow: `airflow db init` (if demonstrating Airflow)
- [ ] Review QUICK_START_LECTURE.md
- [ ] Prepare backup slides/screenshots in case of technical issues

### 1 Hour Before
- [ ] Open required terminals:
  - Terminal 1: MLflow server
  - Terminal 2: Airflow webserver (optional)
  - Terminal 3: Airflow scheduler (optional)
  - Terminal 4: Demo commands
- [ ] Open browser tabs:
  - MLflow UI: http://localhost:5000
  - Airflow UI: http://localhost:8080 (optional)
  - GitHub repo: https://github.com/cloudistic/scalingsystems
- [ ] Test screen sharing
- [ ] Have water ready

## ⏱️ 2-Hour Schedule

| Time | Module | Key Points |
|------|--------|------------|
| 0:00-0:10 | Intro | MLOps challenges, today's tools |
| 0:10-0:40 | DVC | Data versioning, pipelines (30 min) |
| 0:40-1:25 | MLflow | Tracking, comparison, registry (45 min) |
| 1:25-1:55 | Airflow | DAGs, scheduling, monitoring (30 min) |
| 1:55-2:00 | Wrap-up | Q&A, next steps |

## 📂 Essential Files

### To Show in Demos
```
README.md                           # Start here
dvc/README.md                       # DVC tutorial
mlflow/README.md                    # MLflow tutorial
mlflow/demo1_simple.py              # First demo
mlflow/demo2_compare.py             # Model comparison
airflow/README.md                   # Airflow tutorial
airflow/dags/ml_training_pipeline.py # Main demo DAG
QUICK_START_LECTURE.md              # Your guide
```

## 🎬 Command Cheat Sheet

### DVC Quick Demo (10 min)
```bash
cd dvc
mkdir demo && cd demo
git init
dvc init
dvc get https://github.com/iterative/dataset-registry get-started/data.xml -o data.xml
dvc add data.xml
git add data.xml.dvc .gitignore
git commit -m "Track data"
```

### MLflow Demos (30 min)
```bash
# Terminal 1: Start server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: Run demos
cd mlflow
python demo1_simple.py        # 5 min
python demo2_compare.py       # 10 min
python demo3_autolog.py       # 5 min
# Show demo4_registry.py code # 10 min
```

### Airflow Demo (20 min)
```bash
# Show DAG code
cat airflow/dags/ml_training_pipeline.py

# In Airflow UI:
# 1. Navigate to DAGs
# 2. Find ml_training_pipeline
# 3. Show Graph view
# 4. Trigger DAG
# 5. Watch execution
```

## 💡 Key Messages

### DVC (30 min)
- "Git for data" - version large files without bloating Git
- Reproducible pipelines - `dvc repro` runs everything
- Team collaboration - share data via remote storage

**Demo Flow:**
1. Initialize DVC (2 min)
2. Track data file (3 min)
3. Show pipeline concept (5 min)
4. Run pipeline demo (10 min)
5. Q&A (5 min)

### MLflow (45 min)
- Never lose an experiment again
- Compare models objectively
- Manage model lifecycle

**Demo Flow:**
1. Simple tracking (10 min)
2. Model comparison (15 min)
3. Autologging magic (10 min)
4. Model registry (10 min)

### Airflow (30 min)
- Workflows as code (DAGs)
- Automatic scheduling and retries
- Production-ready ML pipelines

**Demo Flow:**
1. Show hello_world DAG (5 min)
2. Explain ML pipeline DAG (10 min)
3. Run and monitor (10 min)
4. Q&A (5 min)

## 🚨 Common Issues & Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| MLflow UI not loading | Check port 5000, try 5001 |
| Import errors | `pip install <package>` |
| Airflow task failing | Check logs in UI, verify paths |
| DVC command fails | Ensure Git is initialized first |
| Demo too slow | Skip autolog or registry demo |
| Demo too fast | Add more model comparisons |

## 🎤 Engagement Tips

### Questions to Ask
- "Who has lost track of experiments?" (before MLflow)
- "How do you currently version data?" (before DVC)
- "How do you schedule ML jobs?" (before Airflow)

### Make It Interactive
- Show failures first, then solutions
- Ask for suggestions before revealing answers
- Use real-world examples from your experience

### Time Savers
- Pre-run some demos if time is tight
- Have screenshots as backup
- Skip optional sections if running late

## 📊 What to Show in UI

### MLflow UI
- ✅ Experiments list
- ✅ Run comparison
- ✅ Parallel coordinates plot
- ✅ Model registry
- ✅ Artifacts

### Airflow UI
- ✅ DAGs list
- ✅ Graph view
- ✅ Task logs
- ✅ Gantt chart
- ✅ Success/failure indicators

## 🎯 Success Indicators

Participants should:
- [ ] Understand what each tool does
- [ ] Know when to use each tool
- [ ] Feel confident to try it themselves
- [ ] Have access to all materials
- [ ] Know where to get help

## 📝 Post-Lecture

### Immediately After
- [ ] Share repository link
- [ ] Share MLflow server URL (if accessible)
- [ ] Collect feedback
- [ ] Answer remaining questions

### Follow-Up
- [ ] Send summary email with links
- [ ] Share additional resources
- [ ] Offer office hours for questions

## 🔗 Quick Links

- Repository: https://github.com/cloudistic/scalingsystems
- DVC Docs: https://dvc.org/doc
- MLflow Docs: https://mlflow.org/docs
- Airflow Docs: https://airflow.apache.org/docs

## 📞 Emergency Contacts

Have ready:
- IT support contact
- Backup presenter (if available)
- Your own contact info for follow-up

---

## 🎯 Remember

**The Goal:** Inspire participants to use these tools, not overwhelm them.

**The Approach:** Show real problems, then show solutions.

**The Outcome:** Participants leave knowing:
1. What these tools do
2. Why they matter
3. How to get started

**Good luck! You've got this! 🚀**

---

*Keep this card visible during your presentation*
