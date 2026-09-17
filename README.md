# 🏭 AI-Powered Packaging Industry Operations Control Tower

### Anomaly Detection, Predictive Risk Analysis & AI-Powered Root-Cause Investigation

An AI-powered operational monitoring and decision-support system designed for the packaging industry.

The project combines **data analytics, machine learning, RAG, Gemini LLM, AI Agent, FastAPI, Streamlit and real-time simulation** to monitor machine operations, detect anomalies, identify operational risks and assist engineers in investigating possible root causes.

---

## 👨‍💻 Author

**Mukesh N**

B.Sc Computer Science Graduate

Coimbatore, Tamil Nadu, India

### Project Role

AI/ML & Data Analytics Project

---

# 📌 1. Project Overview

In a packaging manufacturing environment, machines continuously generate operational data such as:

- Production
- Downtime
- Idle time
- Performance loss
- Scheduled downtime
- Machine changes
- Alarm events

Manually monitoring all these parameters can make it difficult to identify abnormal machine behavior quickly.

This project provides an **AI-powered Operations Control Tower** that brings machine monitoring, anomaly detection, predictive risk analysis and AI-assisted investigation into a single application.

The system helps answer questions such as:

> Which machine is behaving abnormally?

> Which machines have operational risk?

> What evidence supports the alert?

> What should the engineer investigate?

> What possible causes can be considered?

> What actions should be verified by a human operator?

---

# 🎯 2. Project Objective

The main objectives of this project are:

1. Understand packaging machine operational data.
2. Clean and preprocess the dataset.
3. Perform exploratory data analysis.
4. Create operational features.
5. Detect machine anomalies.
6. Predict future operational risk.
7. Store operational data and ML results in MySQL.
8. Build a RAG knowledge base.
9. Use Gemini LLM for AI-assisted investigation.
10. Build an AI Agent capable of using project tools.
11. Perform root-cause investigation.
12. Expose backend functionality through FastAPI.
13. Build an interactive Streamlit Control Tower.
14. Simulate real-time machine operations.
15. Provide supporting evidence for AI decisions.
16. Keep human verification in the final decision loop.

---

# 📊 3. Dataset

## PIADE — Packaging Industry Anomaly Detection Dataset

Dataset source:

Kaggle:

https://www.kaggle.com/datasets/orvile/packaging-industry-anomaly-detection-dataset

The project uses two main CSV files:

```text
raw_data.csv
sequences_1h_data.csv
