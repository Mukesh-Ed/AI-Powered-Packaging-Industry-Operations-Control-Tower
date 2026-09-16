# ============================================================
# PACKAGING AI OPERATIONS CONTROL TOWER
# FastAPI Backend
#
# Includes:
# 1. Health Check
# 2. Machine API
# 3. Machine List API
# 4. Anomaly API
# 5. Risk API
# 6. RAG Knowledge Retrieval
# 7. Gemini AI
# 8. Root Cause Analysis
# 9. AI Agent + Tools
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import os

from pathlib import Path
from dotenv import load_dotenv

from google import genai

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 2. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Packaging AI Operations Control Tower API",
    description="API for machine anomaly, risk and root cause analysis",
    version="1.0.0"
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

# Project root:
# C:\Packaging_AI_Control_Tower

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data folder
DATA_DIR = PROJECT_ROOT / "Data" / "processed"


# CSV files
FEATURES_FILE = DATA_DIR / "features.csv"

ANOMALY_FILE = DATA_DIR / "anomaly_results.csv"

RISK_FILE = DATA_DIR / "risk_results.csv"


# ============================================================
# 4. LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")


if api_key:

    print("Gemini API key loaded successfully")

else:

    print("WARNING: Gemini API key NOT found")


# ============================================================
# 5. CREATE GEMINI CLIENT
# ============================================================

client = None


if api_key:

    try:

        client = genai.Client(
            api_key=api_key
        )

        print(
            "Gemini client created successfully"
        )

    except Exception as e:

        print(
            "WARNING: Gemini client creation failed"
        )

        print(
            "Reason:",
            str(e)
        )


# ============================================================
# 6. GLOBAL DATA VARIABLES
# ============================================================

features_df = None

anomaly_df = None

risk_df = None


# ============================================================
# 7. LOAD PROJECT DATA
# ============================================================

def load_project_data():

    global features_df
    global anomaly_df
    global risk_df


    # --------------------------------------------------------
    # FEATURES DATA
    # --------------------------------------------------------

    if FEATURES_FILE.exists():

        try:

            features_df = pd.read_csv(
                FEATURES_FILE
            )


            # Clean equipment IDs

            if "equipment_ID" in features_df.columns:

                features_df["equipment_ID"] = (
                    features_df["equipment_ID"]
                    .astype(str)
                    .str.strip()
                )


            print(
                f"Features data loaded: "
                f"{features_df.shape}"
            )


        except Exception as e:

            print(
                "ERROR loading features.csv:"
            )

            print(e)

    else:

        print(
            f"WARNING: Features file not found:"
        )

        print(
            FEATURES_FILE
        )


    # --------------------------------------------------------
    # ANOMALY DATA
    # --------------------------------------------------------

    if ANOMALY_FILE.exists():

        try:

            anomaly_df = pd.read_csv(
                ANOMALY_FILE
            )


            if "equipment_ID" in anomaly_df.columns:

                anomaly_df["equipment_ID"] = (
                    anomaly_df["equipment_ID"]
                    .astype(str)
                    .str.strip()
                )


            print(
                f"Anomaly data loaded: "
                f"{anomaly_df.shape}"
            )


        except Exception as e:

            print(
                "ERROR loading anomaly_results.csv:"
            )

            print(e)

    else:

        print(
            "WARNING: Anomaly file not found:"
        )

        print(
            ANOMALY_FILE
        )


    # --------------------------------------------------------
    # RISK DATA
    # --------------------------------------------------------

    if RISK_FILE.exists():

        try:

            risk_df = pd.read_csv(
                RISK_FILE
            )


            if "equipment_ID" in risk_df.columns:

                risk_df["equipment_ID"] = (
                    risk_df["equipment_ID"]
                    .astype(str)
                    .str.strip()
                )


            print(
                f"Risk data loaded: "
                f"{risk_df.shape}"
            )


        except Exception as e:

            print(
                "ERROR loading risk_results.csv:"
            )

            print(e)

    else:

        print(
            "WARNING: Risk file not found:"
        )

        print(
            RISK_FILE
        )


# Load data when API starts
load_project_data()


# ============================================================
# 8. RAG VARIABLES
# ============================================================

embedding_model = None

chroma_client = None

knowledge_collection = None


# ============================================================
# 9. LOAD RAG KNOWLEDGE BASE
# ============================================================

def load_rag():

    global embedding_model
    global chroma_client
    global knowledge_collection


    try:

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


        print(
            "Embedding model loaded successfully"
        )


        # ----------------------------------------------------
        # Search possible ChromaDB locations
        # ----------------------------------------------------

        possible_paths = [

            PROJECT_ROOT / "data" / "chroma_db",

            PROJECT_ROOT / "Data" / "chroma_db"

        ]


        chroma_path = None


        for path in possible_paths:

            if path.exists():

                chroma_path = path

                break


        if chroma_path is None:

            print(
                "WARNING: ChromaDB folder not found"
            )

            return


        # ----------------------------------------------------
        # Connect to ChromaDB
        # ----------------------------------------------------

        chroma_client = chromadb.PersistentClient(
            path=str(chroma_path)
        )


        # ----------------------------------------------------
        # Connect to collection
        # ----------------------------------------------------

        knowledge_collection = (
            chroma_client.get_collection(
                name="packaging_knowledge"
            )
        )


        print(
            "RAG knowledge base connected"
        )


        print(
            "Knowledge documents:",
            knowledge_collection.count()
        )


    except Exception as e:

        print(
            "WARNING: RAG could not be loaded"
        )

        print(
            "Reason:",
            str(e)
        )


# Load RAG
load_rag()


# ============================================================
# 10. HOME API
# ============================================================

@app.get("/")
def home():

    return {

        "message":
        "Packaging AI Operations Control Tower API is running",

        "version":
        "1.0.0",

        "documentation":
        "/docs"
    }


# ============================================================
# 11. HEALTH CHECK API
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
        "healthy",

        "service":
        "Packaging AI Operations Control Tower",

        "data_loaded": {

            "features":
            features_df is not None,

            "anomaly":
            anomaly_df is not None,

            "risk":
            risk_df is not None

        },

        "ai_services": {

            "gemini":
            client is not None,

            "rag":
            knowledge_collection is not None

        }
    }


# ============================================================
# 12. MACHINES LIST API
# ============================================================

@app.get("/machines")
def get_machines():

    if features_df is None:

        raise HTTPException(

            status_code=500,

            detail="Feature data unavailable"
        )


    if "equipment_ID" not in features_df.columns:

        raise HTTPException(

            status_code=500,

            detail="equipment_ID column not found"
        )


    machines = sorted(

        features_df["equipment_ID"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    return {

        "machine_count":
        len(machines),

        "machines":
        machines
    }


# ============================================================
# 13. MACHINE TOOL
# ============================================================

def machine_tool(
    equipment_id: str
):

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if features_df is None:

        return {

            "error":
            "Machine feature data unavailable"
        }


    # --------------------------------------------------------
    # Check equipment column
    # --------------------------------------------------------

    if "equipment_ID" not in features_df.columns:

        return {

            "error":
            "equipment_ID column not found"
        }


    # --------------------------------------------------------
    # Normalize requested ID
    # --------------------------------------------------------

    requested_id = (

        str(equipment_id)
        .strip()
        .upper()
    )


    # --------------------------------------------------------
    # Normalize dataset IDs
    # --------------------------------------------------------

    machine_ids = (

        features_df["equipment_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # --------------------------------------------------------
    # Filter machine
    # --------------------------------------------------------

    machine_data = features_df[
        machine_ids == requested_id
    ].copy()


    # --------------------------------------------------------
    # Machine not found
    # --------------------------------------------------------

    if machine_data.empty:

        return {

            "error":
            f"Machine {equipment_id} not found"
        }


    # --------------------------------------------------------
    # Important operational columns
    # --------------------------------------------------------

    important_columns = [

        "equipment_ID",

        "interval_start",

        "%production",

        "%downtime",

        "%idle",

        "%performance_loss",

        "%scheduled_downtime",

        "#changes",

        "production_rolling_6h",

        "downtime_rolling_6h",

        "production_change",

        "downtime_change",

        "performance_change",

        "health_score"

    ]


    # --------------------------------------------------------
    # Select columns that actually exist
    # --------------------------------------------------------

    available_columns = [

        column

        for column in important_columns

        if column in machine_data.columns

    ]


    machine_data = machine_data[
        available_columns
    ].copy()


    # --------------------------------------------------------
    # Convert date column
    # --------------------------------------------------------

    if "interval_start" in machine_data.columns:

        machine_data["interval_start"] = pd.to_datetime(

            machine_data["interval_start"],

            errors="coerce"
        )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "equipment_id":
        str(equipment_id),

        "records":
        int(len(machine_data))
    }


    # ========================================================
    # AVERAGE METRICS
    # ========================================================

    if "%production" in machine_data.columns:

        value = machine_data[
            "%production"
        ].mean()

        if pd.notna(value):

            result[
                "average_production"
            ] = float(value)


    if "%downtime" in machine_data.columns:

        value = machine_data[
            "%downtime"
        ].mean()

        if pd.notna(value):

            result[
                "average_downtime"
            ] = float(value)


    if "%idle" in machine_data.columns:

        value = machine_data[
            "%idle"
        ].mean()

        if pd.notna(value):

            result[
                "average_idle"
            ] = float(value)


    if "%performance_loss" in machine_data.columns:

        value = machine_data[
            "%performance_loss"
        ].mean()

        if pd.notna(value):

            result[
                "average_performance_loss"
            ] = float(value)


    if "health_score" in machine_data.columns:

        value = machine_data[
            "health_score"
        ].mean()

        if pd.notna(value):

            result[
                "average_health_score"
            ] = float(value)


    # ========================================================
    # LATEST RECORD
    # ========================================================

    if "interval_start" in machine_data.columns:

        machine_data = machine_data.sort_values(

            "interval_start"
        )


    latest = machine_data.iloc[-1]


    # --------------------------------------------------------
    # Convert latest record safely to JSON
    # --------------------------------------------------------

    latest_data = {}


    for column in machine_data.columns:

        value = latest[column]


        # Missing value
        if pd.isna(value):

            latest_data[column] = None


        # Timestamp
        elif isinstance(
            value,
            pd.Timestamp
        ):

            latest_data[column] = (
                value.isoformat()
            )


        # NumPy / Pandas number
        elif hasattr(
            value,
            "item"
        ):

            latest_data[column] = (
                value.item()
            )


        # Normal Python value
        else:

            latest_data[column] = value


    result[
        "latest_data"
    ] = latest_data


    return result


# ============================================================
# 14. MACHINE API
# ============================================================

@app.get("/machine/{equipment_id}")
def get_machine(
    equipment_id: str
):

    result = machine_tool(
        equipment_id
    )


    if "error" in result:

        raise HTTPException(

            status_code=404,

            detail=result["error"]
        )


    return result


# ============================================================
# 15. ANOMALY TOOL
# ============================================================

def anomaly_tool(
    equipment_id: str
):

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if anomaly_df is None:

        return {

            "error":
            "Anomaly data unavailable"
        }


    if "equipment_ID" not in anomaly_df.columns:

        return {

            "error":
            "equipment_ID column not found in anomaly data"
        }


    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    requested_id = (

        str(equipment_id)
        .strip()
        .upper()
    )


    machine_ids = (

        anomaly_df["equipment_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    data = anomaly_df[
        machine_ids == requested_id
    ].copy()


    # --------------------------------------------------------
    # Not found
    # --------------------------------------------------------

    if data.empty:

        return {

            "error":
            f"No anomaly data found for {equipment_id}"
        }


    # --------------------------------------------------------
    # Count anomalies
    # --------------------------------------------------------

    anomaly_count = 0


    if "is_anomaly" in data.columns:

        anomaly_values = pd.to_numeric(

            data["is_anomaly"],

            errors="coerce"

        ).fillna(0)


        anomaly_count = int(
            anomaly_values.sum()
        )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if anomaly_count > 0:

        status = "ANOMALY DETECTED"

    else:

        status = "NO ANOMALY DETECTED"


    # --------------------------------------------------------
    # Recent records
    # --------------------------------------------------------

    recent_records = data.tail(10)


    # Convert records safely
    recent_records = (
        recent_records
        .replace({float("nan"): None})
        .to_dict(
            orient="records"
        )
    )


    return {

        "equipment_id":
        str(equipment_id),

        "anomaly_count":
        anomaly_count,

        "status":
        status,

        "recent_records":
        recent_records
    }


# ============================================================
# 16. ANOMALY API
# ============================================================

@app.get("/anomaly/{equipment_id}")
def get_anomaly(
    equipment_id: str
):

    result = anomaly_tool(
        equipment_id
    )


    if "error" in result:

        raise HTTPException(

            status_code=404,

            detail=result["error"]
        )


    return result


# ============================================================
# 17. RISK TOOL
# ============================================================

def risk_tool(
    equipment_id: str
):

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if risk_df is None:

        return {

            "error":
            "Risk data unavailable"
        }


    if "equipment_ID" not in risk_df.columns:

        return {

            "error":
            "equipment_ID column not found in risk data"
        }


    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    requested_id = (

        str(equipment_id)
        .strip()
        .upper()
    )


    machine_ids = (

        risk_df["equipment_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    data = risk_df[
        machine_ids == requested_id
    ].copy()


    # --------------------------------------------------------
    # Not found
    # --------------------------------------------------------

    if data.empty:

        return {

            "error":
            f"No risk data found for {equipment_id}"
        }


    # --------------------------------------------------------
    # Count risk predictions
    # --------------------------------------------------------

    risk_count = 0


    if "risk_prediction" in data.columns:

        risk_values = pd.to_numeric(

            data["risk_prediction"],

            errors="coerce"

        ).fillna(0)


        risk_count = int(
            risk_values.sum()
        )


    # --------------------------------------------------------
    # Latest risk level
    # --------------------------------------------------------

    latest_risk_level = "UNKNOWN"


    if "risk_level" in data.columns:

        valid_levels = (

            data["risk_level"]
            .dropna()
        )


        if not valid_levels.empty:

            latest_risk_level = str(

                valid_levels.iloc[-1]
            )


    # --------------------------------------------------------
    # Risk status
    # --------------------------------------------------------

    if risk_count > 0:

        status = "RISK DETECTED"

    else:

        status = "NO HIGH-RISK PREDICTION"


    # --------------------------------------------------------
    # Recent records
    # --------------------------------------------------------

    recent_records = data.tail(10)


    recent_records = (

        recent_records
        .replace({float("nan"): None})
        .to_dict(
            orient="records"
        )
    )


    return {

        "equipment_id":
        str(equipment_id),

        "risk_count":
        risk_count,

        "risk_status":
        status,

        "latest_risk_level":
        latest_risk_level,

        "recent_records":
        recent_records
    }


# ============================================================
# 18. RISK API
# ============================================================

@app.get("/risk/{equipment_id}")
def get_risk(
    equipment_id: str
):

    result = risk_tool(
        equipment_id
    )


    if "error" in result:

        raise HTTPException(

            status_code=404,

            detail=result["error"]
        )


    return result


# ============================================================
# 19. RAG TOOL
# ============================================================

def rag_tool(
    query: str,
    top_k: int = 5
):

    # --------------------------------------------------------
    # Check RAG
    # --------------------------------------------------------

    if (
        embedding_model is None
        or knowledge_collection is None
    ):

        return {

            "status":
            "RAG unavailable",

            "context":
            ""
        }


    try:

        # ----------------------------------------------------
        # Create embedding
        # ----------------------------------------------------

        query_embedding = (
            embedding_model.encode(
                [query]
            )[0]
        )


        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        results = (
            knowledge_collection.query(

                query_embeddings=[
                    query_embedding.tolist()
                ],

                n_results=top_k
            )
        )


        # ----------------------------------------------------
        # Get documents
        # ----------------------------------------------------

        documents = results.get(
            "documents",
            [[]]
        )[0]


        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]


        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context_parts = []


        for i, document in enumerate(
            documents
        ):

            source = "Unknown"


            if i < len(metadatas):

                if metadatas[i]:

                    source = metadatas[i].get(

                        "source",

                        "Unknown"
                    )


            context_parts.append(

                f"""
SOURCE: {source}

CONTENT:
{document}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        return {

            "status":
            "RAG retrieval successful",

            "context":
            context
        }


    except Exception as e:

        return {

            "status":
            "RAG retrieval failed",

            "context":
            "",

            "error":
            str(e)
        }


# ============================================================
# 20. RCA REQUEST MODEL
# ============================================================

class RCARequest(BaseModel):

    equipment_id: str

    question: str


# ============================================================
# 21. AI AGENT
# ============================================================

def run_ai_agent(

    equipment_id: str,

    question: str

):

    # ========================================================
    # TOOL 1
    # MACHINE DATA
    # ========================================================

    machine_info = machine_tool(

        equipment_id
    )


    # ========================================================
    # TOOL 2
    # ANOMALY DATA
    # ========================================================

    anomaly_info = anomaly_tool(

        equipment_id
    )


    # ========================================================
    # TOOL 3
    # RISK DATA
    # ========================================================

    risk_info = risk_tool(

        equipment_id
    )


    # ========================================================
    # TOOL 4
    # RAG
    # ========================================================

    rag_query = f"""

Machine ID:
{equipment_id}

User question:
{question}

Investigate:
machine operation,
downtime,
alarms,
anomalies,
and operational risk.

What should be investigated?
"""


    rag_info = rag_tool(

        rag_query,

        top_k=5
    )


    knowledge = rag_info.get(

        "context",

        ""
    )


    # ========================================================
    # MACHINE CHECK
    # ========================================================

    if "error" in machine_info:

        return {

            "equipment_id":
            equipment_id,

            "status":
            "Machine not found",

            "machine_data":
            machine_info
        }


    # ========================================================
    # GEMINI CHECK
    # ========================================================

    if client is None:

        return {

            "equipment_id":
            equipment_id,

            "status":
            "Gemini is not configured",

            "machine_data":
            machine_info,

            "anomaly_data":
            anomaly_info,

            "risk_data":
            risk_info,

            "rag_knowledge":
            knowledge
        }


    # ========================================================
    # CREATE AI PROMPT
    # ========================================================

    prompt = f"""

You are an AI Operations Control Tower Agent
for a packaging industry.

Your job is to investigate machine operational
problems using available evidence.

================================================
USER QUESTION
================================================

{question}


================================================
EQUIPMENT
================================================

{equipment_id}


================================================
MACHINE OPERATIONAL DATA
================================================

{machine_info}


================================================
ANOMALY DETECTION RESULTS
================================================

{anomaly_info}


================================================
RISK PREDICTION RESULTS
================================================

{risk_info}


================================================
RETRIEVED PROJECT KNOWLEDGE
================================================

{knowledge}


================================================
IMPORTANT RULES
================================================

1. Analyze the available evidence carefully.

2. Clearly separate observed evidence
   from possible causes.

3. Do NOT invent the meaning of alarm codes.

4. Do NOT claim a possible cause is
   a confirmed root cause.

5. Do NOT invent missing machine data.

6. Use retrieved project knowledge
   where relevant.

7. If the knowledge base does not contain
   enough information, say:

   "Insufficient information in the knowledge base."

8. Possible causes are hypotheses only.

9. Provide practical investigation steps.

10. Physical machine maintenance or intervention
    requires human verification.

11. Do not claim that the AI physically controlled
    or repaired the machine.


================================================
RETURN FORMAT
================================================

Machine:

Incident Summary:

Observed Evidence:

Anomaly Status:

Risk Status:

Possible Root Cause:

Supporting Evidence:

Confidence:

Recommended Investigation:

Recommended Action:

Human Verification:

"""


    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        response = client.models.generate_content(

            model="gemini-3.1-flash-lite",

            contents=prompt
        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "equipment_id":
            equipment_id,

            "question":
            question,

            "status":
            "RCA completed",

            "machine_data":
            machine_info,

            "anomaly_data":
            anomaly_info,

            "risk_data":
            risk_info,

            "rag_status":
            rag_info.get(
                "status",
                "Unknown"
            ),

            "root_cause_analysis":
            response.text
        }


    except Exception as e:

        return {

            "equipment_id":
            equipment_id,

            "status":
            "Gemini request failed",

            "error":
            str(e),

            "machine_data":
            machine_info,

            "anomaly_data":
            anomaly_info,

            "risk_data":
            risk_info,

            "rag_knowledge":
            knowledge
        }


# ============================================================
# 22. RCA API
# ============================================================

@app.post("/rca")
def root_cause_analysis(

    request: RCARequest

):

    result = run_ai_agent(

        equipment_id=
        request.equipment_id,

        question=
        request.question
    )


    return result


# ============================================================
# END OF MAIN.PY
# ============================================================