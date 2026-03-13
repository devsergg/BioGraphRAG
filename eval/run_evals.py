"""
LangSmith evaluation runner.

Prerequisites:
  1. FastAPI server is running: uvicorn app.main:app --reload --port 8000
  2. ground_truth.py is populated with actual NCT IDs and expected answers
  3. LANGCHAIN_API_KEY is set in .env

Usage:
    python eval/run_evals.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from langsmith import Client
from langsmith.evaluation import evaluate
from eval.ground_truth import GROUND_TRUTH_QA

API_BASE = "http://localhost:8000/api"
DATASET_NAME = "pain-graphrag-eval-v1"


def query_system(inputs: dict) -> dict:
    """Call the live FastAPI pipeline and return the full response."""
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": inputs["question"]},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def context_precision_evaluator(run, example) -> dict:
    """
    Custom evaluator: what fraction of retrieved NCT IDs overlap with expected ones?
    Score: 0.0 – 1.0
    """
    expected_ncts = set(example.outputs.get("relevant_nct_ids", []))
    if not expected_ncts:
        return {"key": "context_precision", "score": 1.0}

    retrieved_ncts = set(
        src.get("nct_id", "") for src in run.outputs.get("sources", [])
    )
    overlap = expected_ncts & retrieved_ncts
    score = len(overlap) / len(expected_ncts)
    return {"key": "context_precision", "score": score}


def create_or_get_dataset(client: Client) -> str:
    """Create the LangSmith dataset from ground truth, or return existing one."""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"  Using existing dataset: {DATASET_NAME}")
        return dataset.id
    except Exception:
        print(f"  Creating new dataset: {DATASET_NAME}")
        dataset = client.create_dataset(DATASET_NAME)
        examples = [
            {
                "inputs": {"question": qa["question"]},
                "outputs": {
                    "answer": qa["expected_answer"],
                    "relevant_nct_ids": qa["relevant_nct_ids"],
                    "category": qa["category"],
                },
            }
            for qa in GROUND_TRUTH_QA
        ]
        client.create_examples(dataset_id=dataset.id, examples=examples)
        return dataset.id


def run_evaluations():
    print("=" * 60)
    print("Biotech GraphRAG — LangSmith Evaluation Run")
    print("=" * 60)

    client = Client()
    dataset_id = create_or_get_dataset(client)

    print("\nRunning evaluations (this will call the live API 30 times)...")
    results = evaluate(
        query_system,
        data=dataset_id,
        evaluators=[context_precision_evaluator],
        experiment_prefix="pain-graphrag-baseline",
        client=client,
        metadata={"version": "0.1.0", "retrieval": "hybrid-pinecone-neo4j"},
    )

    print("\nEvaluation complete.")
    print(f"Results: {results.url}")


if __name__ == "__main__":
    run_evaluations()
