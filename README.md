# 🏭 AI-Powered Packaging Industry Operations Control Tower

### Anomaly Detection | Predictive Risk Analysis | RAG | Gemini AI | AI Agent | Root-Cause Investigation | FastAPI | Streamlit | Real-Time Simulation

---

## 📌 Project Overview

The **AI-Powered Packaging Industry Operations Control Tower** is an AI-driven operational monitoring and decision-support system designed for packaging industry machine operations.

The system combines data analytics, machine learning, Retrieval-Augmented Generation (RAG), Generative AI, an AI Agent, REST APIs, Streamlit visualization, and real-time simulation.

The main objective is to transform raw machine-operation data into useful operational insights.

The system can monitor:

- Production
- Downtime
- Idle time
- Performance loss
- Scheduled downtime
- Machine changes
- Machine health
- Anomalies
- Operational risk
- Machine alarms
- AI-assisted investigation guidance

---

## 🎯 Project Objective

The main objectives of this project are:

1. Understand packaging machine operational data.
2. Clean and prepare raw operational datasets.
3. Perform exploratory data analysis.
4. Create useful operational features.
5. Calculate a project-defined machine health score.
6. Detect abnormal machine behavior.
7. Predict project-defined future operational risk.
8. Store operational information in MySQL.
9. Build a domain knowledge base.
10. Implement semantic document retrieval.
11. Integrate Gemini AI.
12. Build an AI Agent using multiple tools.
13. Perform root-cause investigation.
14. Expose functionality through FastAPI.
15. Build an interactive Streamlit Control Tower.
16. Simulate real-time machine operations.

---

# 🚨 Industry Problem

Packaging industries continuously monitor machines and production lines.

Machine problems can result in:

- Production reduction
- Increased downtime
- Performance loss
- Repeated alarms
- Operational inefficiency
- Maintenance requirements
- Potential production losses

When operational information is distributed across different sources, investigation can require manual analysis.

The project addresses this problem by bringing operational data, machine-learning results, knowledge retrieval, and AI-assisted investigation into one control-tower workflow.

---

# 💡 Proposed Solution

The proposed solution is an **AI Operations Control Tower**.

The Control Tower receives machine-operation data and processes it through multiple analytical and AI components.

The system follows this workflow:

```text
Raw Machine Data
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
MySQL Database
       ↓
Anomaly Detection
       ↓
Risk Prediction
       ↓
RAG Knowledge Base
       ↓
Gemini AI
       ↓
AI Agent
       ↓
Root-Cause Investigation
       ↓
FastAPI
       ↓
Streamlit Control Tower
       ↓
Real-Time Simulation
```

---

# 📊 Dataset

This project uses the:

**PIADE – Packaging Industry Anomaly Detection Dataset**

Dataset source:

https://www.kaggle.com/datasets/orvile/packaging-industry-anomaly-detection-dataset

The project works with operational machine data and hourly machine-level information.

---

# 📁 Dataset Files

The project uses two main datasets:

```text
raw_data.csv
sequences_1h_data.csv
```

The raw dataset contains approximately:

```text
429,394 rows
10 columns
```

The hourly dataset contains approximately:

```text
23,376 rows
164 columns
```

---

# 🧾 Raw Dataset Columns

Important raw dataset fields include:

```text
interval_start
equipment_ID
alarm
type
start
end
elapsed
pi
po
speed
```

These fields provide information about equipment operation, events, alarms, timing, and machine performance.

---

# 📈 Hourly Operational Data

The hourly dataset contains operational metrics including:

```text
%production
%downtime
%idle
%performance_loss
%scheduled_downtime
#changes
```

Additional alarm and operational features are also available.

---

# ⚠️ Important Data Understanding

The operational percentage fields in the dataset are represented as proportions.

For example:

```text
0.80
0.50
0.20
```

represent proportional values rather than:

```text
80
50
20
```

Therefore, the project does not incorrectly multiply or compare these fields using 0–100 thresholds during the ML workflow.

---

# 🏗️ Project Architecture

```text
                  ┌──────────────────────┐
                  │    PIADE Dataset     │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   Data Processing    │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Feature Engineering  │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │       MySQL          │
                  └──────────┬───────────┘
                             ↓
             ┌───────────────┴───────────────┐
             ↓                               ↓
   ┌───────────────────┐          ┌───────────────────┐
   │ Anomaly Detection │          │  Risk Prediction  │
   └─────────┬─────────┘          └─────────┬─────────┘
             └───────────────┬───────────────┘
                             ↓
                  ┌──────────────────────┐
                  │   RAG Knowledge      │
                  │       Base           │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │     Gemini AI        │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │      AI Agent        │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Root-Cause Analysis  │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │       FastAPI        │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Streamlit Dashboard  │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Real-Time Simulation │
                  └──────────────────────┘
```

---

# 🗂️ Project Structure

```text
Packaging_AI_Control_Tower/
│
├── api/
│   └── main.py
│
├── Data/
│   ├── raw/
│   ├── processed/
│   └── chroma_db/
│
├── documents/
│   ├── machine_manuals/
│   │   └── packaging_machine_operations.txt
│   │
│   ├── alarm_guides/
│   │   └── packaging_alarm_investigation.txt
│   │
│   ├── maintenance_guides/
│   │   └── packaging_maintenance_guidelines.txt
│   │
│   └── incident_reports/
│       └── packaging_incident_investigation.txt
│
├── models/
│   ├── anomaly_model.pkl
│   └── risk_model.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_risk_prediction.ipynb
│   ├── 07_evaluation.ipynb
│   ├── 08_mysql_database.ipynb
│   └── 09_rag_knowledge_base.ipynb
│
├── src/
│   └── simulation/
│       └── realtime_simulator.py
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🔹 Phase 1 — Data Understanding

Notebook:

```text
01_data_understanding.ipynb
```

The first phase focuses on understanding the dataset.

Activities include:

- Loading CSV files
- Checking dataset shape
- Viewing sample records
- Checking column names
- Inspecting data types
- Checking missing values
- Checking duplicate records
- Identifying equipment IDs
- Studying alarm information
- Studying operational fields
- Generating descriptive statistics

The objective is to understand the structure before performing data transformation.

---

# 🔹 Phase 2 — Data Cleaning

Notebook:

```text
02_data_cleaning.ipynb
```

The cleaning phase prepares the data for analytics and ML.

Operations include:

- Removing duplicate records
- Parsing date/time columns
- Converting numerical fields
- Handling missing values
- Preparing consistent datasets
- Saving cleaned datasets

Output files include:

```text
Data/processed/raw_cleaned.csv
Data/processed/hourly_cleaned.csv
```

---

# 🔹 Phase 3 — Exploratory Data Analysis

Notebook:

```text
03_eda.ipynb
```

EDA is used to understand operational patterns.

Important analysis includes:

- Average production by machine
- Average downtime by machine
- Production versus downtime
- Performance loss analysis
- Correlation analysis
- Machine-level comparison
- Operational distributions

EDA helps identify machines and patterns that require further investigation.

---

# 🔹 Phase 4 — Feature Engineering

Notebook:

```text
04_feature_engineering.ipynb
```

Feature engineering creates additional variables useful for ML.

The project creates:

```text
production_rolling_6h
downtime_rolling_6h
production_change
downtime_change
performance_change
health_score
```

---

# ❤️ Machine Health Score

A project-defined health score is calculated from operational metrics.

The formula is:

```text
Health Score =
100
- Downtime
- Idle
- Performance Loss
```

The result is clipped between:

```text
0 and 100
```

The health score is a project-created metric.

It is not a field directly provided by the PIADE dataset.

---

# 🔹 Phase 5 — Anomaly Detection

Notebook:

```text
05_anomaly_detection.ipynb
```

The project uses:

```text
Isolation Forest
```

Isolation Forest is an unsupervised machine-learning technique.

It is used because the dataset does not provide a simple supervised ground-truth anomaly label for this project.

---

# 🤖 Anomaly Features

The anomaly model uses operational variables such as:

```text
%production
%downtime
%idle
%performance_loss
%scheduled_downtime
#changes
```

Model configuration:

```text
n_estimators = 200
contamination = 0.05
random_state = 42
```

---

# 📤 Anomaly Outputs

The system generates:

```text
anomaly_prediction
anomaly_score
is_anomaly
```

Model:

```text
models/anomaly_model.pkl
```

Results:

```text
Data/processed/anomaly_results.csv
```

---

# ⚠️ Anomaly Interpretation

An anomaly means that the operational pattern was identified as unusual by the Isolation Forest model.

It does not automatically mean that a machine has physically failed.

Further investigation is required.

---

# 🔹 Phase 6 — Predictive Risk Analysis

Notebook:

```text
06_risk_prediction.ipynb
```

The risk model is designed to predict future operational deterioration.

Because the dataset does not provide a direct machine-failure-within-24-hours ground-truth label for this project, a transparent project-defined future-risk target is created.

---

# 🎯 Future Risk Target

The project calculates future operational values using future observations.

Risk conditions include:

```text
Future production below 25th percentile
Future downtime above 75th percentile
Future performance loss above 75th percentile
```

The number of conditions is counted.

When at least two conditions are satisfied:

```text
future_risk = 1
```

Otherwise:

```text
future_risk = 0
```

This is a project-defined operational-risk target.

---

# 📊 Risk Prediction Model

The project uses:

```text
XGBoost Classifier
```

Features include:

```text
%production
%downtime
%idle
%performance_loss
%scheduled_downtime
#changes
production_rolling_6h
downtime_rolling_6h
production_change
downtime_change
performance_change
health_score
```

---

# ⚙️ XGBoost Configuration

```text
n_estimators = 300
max_depth = 5
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

A chronological 80/20 split is used to reduce temporal leakage.

---

# 🚦 Risk Levels

The project maps predicted probability to:

```text
CRITICAL
HIGH
MEDIUM
LOW
```

Project thresholds:

```text
>= 80%  → CRITICAL
>= 60%  → HIGH
>= 30%  → MEDIUM
< 30%   → LOW
```

These are project-defined dashboard categories and are not manufacturer severity standards.

---

# 📤 Risk Outputs

Generated files include:

```text
Data/processed/risk_results.csv
Data/processed/machine_risk_summary.csv
Data/processed/risk_feature_importance.csv
models/risk_model.pkl
```

---

# 🔹 Phase 7 — Model Evaluation

Notebook:

```text
07_evaluation.ipynb
```

The risk model is evaluated using:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Confusion Matrix
```

Additional evaluation includes:

- ROC curve
- Precision-Recall curve
- Feature importance
- False-negative analysis

---

# ⚠️ Evaluation Limitation

Because the target is project-defined future operational risk, the evaluation measures how well the model predicts that defined target.

It should not be described as proven real-world machine-failure prediction unless validated using actual failure labels.

---

# 🔹 Phase 8 — MySQL Database

Notebook:

```text
08_mysql_database.ipynb
```

MySQL is used as the operational database layer.

Database:

```text
packaging_control_tower
```

The project uses database tables for operational information, anomaly results, and risk results.

---

# 🗄️ Main MySQL Tables

```text
machine_operations
machine_anomalies
machine_risk
```

---

# 📋 machine_operations

Stores operational information such as:

```text
equipment_ID
interval_start
production
downtime
idle
performance_loss
scheduled_downtime
changes_count
health_score
```

---

# 📋 machine_anomalies

Stores:

```text
equipment_ID
interval_start
anomaly_prediction
anomaly_score
is_anomaly
```

---

# 📋 machine_risk

Stores:

```text
equipment_ID
interval_start
future_risk
risk_prediction
risk_probability
risk_level
```

---

# 🔹 Phase 9 — RAG Knowledge Base

Notebook:

```text
09_rag_knowledge_base.ipynb
```

RAG stands for:

```text
Retrieval-Augmented Generation
```

The RAG system provides project-specific knowledge to the Generative AI component.

---

# 📚 Knowledge Documents

The project contains knowledge documents covering:

```text
Machine Operations
Alarm Investigation
Maintenance Guidelines
Incident Investigation
```

These documents provide general project knowledge for operational investigation.

They are not manufacturer-specific manuals.

---

# 🔎 RAG Workflow

```text
Knowledge Documents
        ↓
Document Loading
        ↓
Text Cleaning
        ↓
Text Chunking
        ↓
Embeddings
        ↓
ChromaDB
        ↓
Semantic Search
        ↓
Relevant Context
        ↓
Gemini AI
```

---

# 🧠 Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

through:

```text
Sentence Transformers
```

The embedding dimension used is:

```text
384
```

---

# 🗃️ ChromaDB

ChromaDB is used as the vector database.

The collection is:

```text
packaging_knowledge
```

The vector store allows semantic retrieval of relevant project knowledge based on the user's question.

---

# 🔹 Phase 10 — Gemini AI

Gemini is used as the Generative AI component.

It receives:

```text
User Question
+
Retrieved Knowledge
+
Operational Evidence
```

and generates an investigation-oriented response.

---

# 🔐 Gemini API Key

The API key is stored in:

```text
.env
```

Example:

```text
GEMINI_API_KEY=YOUR_API_KEY
```

The API key must not be committed to GitHub.

The `.env` file is included in `.gitignore`.

---

# 🧠 Gemini Instructions

The AI is instructed to:

- Use retrieved knowledge
- Separate facts from possible causes
- Avoid inventing alarm meanings
- Avoid presenting possible causes as confirmed causes
- Identify insufficient information
- Recommend investigation steps
- Require human verification before physical intervention

---

# 🔹 Phase 11 — AI Agent

The AI Agent coordinates different tools and information sources.

The agent can work with:

```text
Machine Data Tool
Anomaly Detection Tool
Risk Prediction Tool
RAG Search Tool
Database Tool
```

---

# 🤖 AI Agent Workflow

```text
User Question
      ↓
AI Agent
      ↓
Machine Data
      ↓
Anomaly Result
      ↓
Risk Result
      ↓
RAG Search
      ↓
Evidence Collection
      ↓
Gemini Reasoning
      ↓
Investigation Response
```

The AI Agent acts as an orchestration layer rather than relying only on the language model.

---

# 🔹 Phase 12 — Root-Cause Investigation

The Root-Cause Analysis component combines:

```text
Operational Data
+
Anomaly Detection
+
Risk Prediction
+
RAG Knowledge
+
Gemini AI
```

The system produces a structured investigation.

---

# 🧾 RCA Response Format

The response is structured as:

```text
Incident Summary
Evidence
Possible Root Cause
Confidence
Recommended Investigation
Human Verification
```

---

# ⚠️ RCA Safety Principle

The system does not automatically claim that a possible cause is a confirmed physical root cause.

The AI provides investigation assistance.

Physical machine maintenance or intervention requires human verification.

---

# 🔹 Phase 13 — FastAPI

File:

```text
api/main.py
```

FastAPI provides a REST API layer between the backend services and the user interface.

---

# 🌐 API Endpoints

The project includes:

```text
GET /health
GET /machines
GET /machine/{machine_id}
GET /anomaly/{machine_id}
GET /risk/{machine_id}
POST /rca
```

---

# ❤️ Health Endpoint

```text
GET /health
```

Used to check whether the API service is running.

---

# 🏭 Machines Endpoint

```text
GET /machines
```

Returns available machine/equipment information.

---

# 🔍 Machine Endpoint

```text
GET /machine/{machine_id}
```

Returns operational information for a selected machine.

Example:

```text
/machine/s_1
```

---

# 🚨 Anomaly Endpoint

```text
GET /anomaly/{machine_id}
```

Returns anomaly-related information for the selected machine.

---

# 🚦 Risk Endpoint

```text
GET /risk/{machine_id}
```

Returns risk prediction information.

---

# 🧠 RCA Endpoint

```text
POST /rca
```

Used to request AI-assisted root-cause investigation.

---

# 🔹 Phase 14 — Streamlit Control Tower

File:

```text
streamlit_app/app.py
```

Streamlit provides the interactive frontend.

The dashboard is designed as an operational Control Tower.

---

# 📊 Dashboard Features

The dashboard provides:

- Machine selection
- Production monitoring
- Downtime monitoring
- Idle monitoring
- Performance-loss monitoring
- Health score
- Anomaly status
- Risk status
- AI investigation
- Supporting evidence
- Real-time simulation

---

# 📌 KPI Monitoring

Important KPIs include:

```text
Production
Downtime
Idle
Performance Loss
Health Score
```

These KPIs help users quickly understand machine status.

---

# 🚨 Operational Alerts

The dashboard can highlight:

```text
Anomaly Detected
High Risk
Critical Risk
Operational Degradation
```

The alerts are based on project-defined ML outputs and thresholds.

---

# 🔹 Phase 15 — Real-Time Simulation

File:

```text
src/simulation/realtime_simulator.py
```

The real-time simulation replays historical operational records sequentially.

This provides a demonstration of how the Control Tower could process incoming machine observations.

---

# ⏱️ Simulation Workflow

```text
Historical Data
      ↓
Select Machine
      ↓
Read Next Record
      ↓
Update Dashboard
      ↓
Check Anomaly
      ↓
Check Risk
      ↓
Generate Alert
      ↓
AI Investigation
      ↓
Display Result
```

---

# 🔄 Real-Time Dashboard

The Streamlit dashboard can use the simulation to display machine records progressively.

This creates a real-time monitoring experience using historical data as the simulation source.

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | EDA visualization |
| Scikit-learn | Anomaly detection |
| XGBoost | Risk prediction |
| MySQL | Database |
| ChromaDB | Vector database |
| Sentence Transformers | Embeddings |
| Gemini AI | Generative AI |
| FastAPI | REST API |
| Streamlit | Dashboard |
| Git | Version control |
| GitHub | Project repository |

---

# 📦 Installation

Create and activate a Python virtual environment.

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file in the project root.

```text
GEMINI_API_KEY=YOUR_API_KEY
```

Never upload the actual API key to GitHub.

---

# ▶️ Run FastAPI

From the project root:

```bash
uvicorn api.main:app --reload
```

The API can then be accessed through the local FastAPI server.

---

# ▶️ Run Streamlit

From the project root:

```bash
streamlit run streamlit_app/app.py
```

The Streamlit application opens the Control Tower dashboard.

---

# 🧪 Testing

The project can be tested at multiple levels.

Testing areas include:

```text
Data Processing
Feature Engineering
ML Models
RAG Retrieval
Gemini Integration
FastAPI Endpoints
Streamlit Dashboard
Real-Time Simulation
```

---

# 🔐 Security

Important security practices:

- API keys are stored in `.env`
- `.env` is excluded using `.gitignore`
- Credentials should never be committed
- Production deployments should use secure secrets management
- Physical machine actions require human verification

---

# 📈 End-to-End Project Workflow

```text
1. Load PIADE Dataset
2. Understand Dataset
3. Clean Data
4. Perform EDA
5. Engineer Features
6. Calculate Health Score
7. Detect Anomalies
8. Predict Operational Risk
9. Evaluate Models
10. Store Data in MySQL
11. Create Knowledge Documents
12. Generate Embeddings
13. Store Embeddings in ChromaDB
14. Retrieve Relevant Knowledge
15. Connect Gemini AI
16. Build AI Agent
17. Perform RCA
18. Create FastAPI
19. Build Streamlit Dashboard
20. Connect Real-Time Simulation
```

---

# 🧠 Overall AI Workflow

```text
Machine Data
      ↓
Machine Learning
      ↓
Anomaly / Risk
      ↓
Evidence
      ↓
Knowledge Retrieval
      ↓
RAG Context
      ↓
Gemini AI
      ↓
AI Agent
      ↓
Root-Cause Investigation
      ↓
Recommendation
      ↓
Human Verification
```

---

# 👥 Intended Users

The Control Tower concept can support users involved in operational monitoring and investigation, such as:

- Operations teams
- Production teams
- Maintenance teams
- Supervisors
- Manufacturing analysts
- Data analysts
- Operations managers

The system is designed as a decision-support tool.

---

# 🌟 Key Project Features

### 1. Machine Monitoring

Monitor machine-level operational metrics.

### 2. Anomaly Detection

Identify unusual operational patterns using Isolation Forest.

### 3. Risk Prediction

Predict project-defined future operational risk using XGBoost.

### 4. Knowledge Retrieval

Retrieve relevant project knowledge using semantic search.

### 5. Generative AI

Use Gemini to generate structured investigation responses.

### 6. AI Agent

Coordinate multiple tools and information sources.

### 7. Root-Cause Investigation

Combine machine evidence, ML results, and retrieved knowledge.

### 8. REST API

Expose backend functionality using FastAPI.

### 9. Interactive Dashboard

Display operational information using Streamlit.

### 10. Real-Time Simulation

Replay historical machine data as a real-time demonstration.

---

# ⚠️ Project Limitations

The project has several important limitations.

1. The anomaly model is unsupervised.
2. The dataset does not provide a direct ground-truth failure label for this project.
3. The future-risk target is project-defined.
4. Risk thresholds are project-defined.
5. Health score is project-defined.
6. RAG documents are project knowledge documents.
7. Alarm meanings should not be invented when unavailable.
8. AI-generated causes are investigation hypotheses.
9. Physical intervention requires human verification.
10. Real-time simulation replays historical data rather than receiving live industrial sensor streams.

---

# 🚀 Future Enhancements

Possible future improvements include:

- Live IoT sensor integration
- Real-time industrial data streaming
- Kafka integration
- MQTT integration
- Advanced predictive maintenance
- Actual failure labels
- Time-series forecasting
- Automated alert notifications
- Advanced agent planning
- Role-based access control
- Cloud deployment
- Docker deployment
- Kubernetes deployment
- Monitoring and logging
- Model retraining pipelines
- Production-grade security

---

# 🎓 Viva Explanation

## What is this project?

This is an AI-powered Packaging Industry Operations Control Tower that monitors machine operations, detects anomalies, predicts operational risk, retrieves relevant knowledge, and provides AI-assisted root-cause investigation.

---

## Why did you choose this project?

Packaging industries generate large amounts of operational data. The project demonstrates how data analytics, machine learning, Generative AI, RAG, and APIs can be combined to support operational monitoring and investigation.

---

## Why use Isolation Forest?

Isolation Forest is an unsupervised anomaly-detection algorithm that can identify unusual observations without requiring a predefined anomaly label.

---

## Why use XGBoost?

XGBoost is used for the project-defined future operational-risk classification task and can model nonlinear relationships between operational features.

---

## Why use RAG?

RAG allows the AI system to retrieve relevant project knowledge before generating an answer.

This helps ground the response in the available knowledge base.

---

## Why use ChromaDB?

ChromaDB stores embeddings and enables semantic similarity search over the project knowledge documents.

---

## Why use Gemini?

Gemini is used as the Generative AI layer for reasoning over retrieved knowledge and operational evidence.

---

## Why use FastAPI?

FastAPI provides REST endpoints so that backend functionality can be accessed by applications such as the Streamlit dashboard.

---

## Why use Streamlit?

Streamlit allows the project to present operational data, ML results, alerts, and AI investigation results through an interactive web interface.

---

# 🏁 Final Project Flow

```text
                    🏭
             PACKAGING DATA
                    │
                    ▼
             DATA PROCESSING
                    │
                    ▼
          FEATURE ENGINEERING
                    │
                    ▼
             ┌─────────────┐
             │   MACHINE   │
             │   LEARNING  │
             └──────┬──────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      ANOMALY               RISK
     DETECTION            PREDICTION
          │                   │
          └─────────┬─────────┘
                    ▼
              OPERATIONAL
                 EVIDENCE
                    │
                    ▼
               RAG SEARCH
                    │
                    ▼
               GEMINI AI
                    │
                    ▼
                AI AGENT
                    │
                    ▼
             ROOT-CAUSE
             INVESTIGATION
                    │
                    ▼
                FASTAPI
                    │
                    ▼
              STREAMLIT
              CONTROL TOWER
                    │
                    ▼
           REAL-TIME SIMULATION
```

---

# 📌 Project Summary

The **AI-Powered Packaging Industry Operations Control Tower** demonstrates an end-to-end AI and analytics workflow for packaging machine operations.

The project combines:

```text
Data Engineering
+
Data Analytics
+
Machine Learning
+
Anomaly Detection
+
Risk Prediction
+
MySQL
+
RAG
+
Vector Database
+
Generative AI
+
AI Agent
+
Root-Cause Investigation
+
FastAPI
+
Streamlit
+
Real-Time Simulation
```

The final system provides a unified environment for monitoring machine operations and supporting operational investigation.

---

# 👨‍💻 Author

## Mukesh N

**B.Sc Computer Science Graduate**

**VLB Janakiammal College of Arts and Science, Coimbatore**

### Areas of Interest

```text
Python
Data Analytics
Machine Learning
Artificial Intelligence
Generative AI
SQL
Power BI
FastAPI
Streamlit
```

---

# ⭐ Project

**AI-Powered Packaging Industry Operations Control Tower**

### Subtitle

**Anomaly Detection, Production Loss Analysis, Predictive Risk & AI Root-Cause Assistant**

---

## 📚 Dataset

PIADE – Packaging Industry Anomaly Detection Dataset

https://www.kaggle.com/datasets/orvile/packaging-industry-anomaly-detection-dataset

---

## 🔖 Keywords

```text
Python
Data Analytics
Machine Learning
AI
Generative AI
RAG
LLM
AI Agent
Anomaly Detection
Isolation Forest
XGBoost
Predictive Analytics
Root Cause Analysis
ChromaDB
MySQL
FastAPI
Streamlit
Real-Time Simulation
Packaging Industry
Operations Control Tower
```

---

## 📄 License

This project is created for educational, academic, and portfolio purposes.

The dataset remains subject to its original source and licensing terms.

---

# 🙌 Thank You

**Built by Mukesh N**

**AI-Powered Packaging Industry Operations Control Tower**
'''


