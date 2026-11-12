# RAGAS retriever test: did the retriever fetch the right notes?
# Run it from the project folder:  python ragas_evals/retriever_ragas.py

import csv
import json
import math
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall

sys.path.append(".")  # lets Python find our src folder
from src.rag_chain import get_retriever

load_dotenv()

# How many chunks the retriever returns. Try 2, then 8.
K = 4

# Q030 to Q036 are rude, tricky or out-of-scope messages, so we skip them
SKIP_IDS = ["Q030", "Q031", "Q032", "Q033", "Q034", "Q035", "Q036"]


def average(scores):
    valid = []
    for score in scores:
        if not math.isnan(score):  # RAGAS gives NaN when it can't score a row
            valid.append(score)
    return round(sum(valid) / len(valid), 2)


def run():

    # Step 1: Read the Goldens (questions + expected answers)
    with open("goldens/goldens.json", encoding="utf-8") as file:
        goldens = json.load(file)

    # Step 2: Set up the judge, and the two retriever metrics
    judge = llm_factory("gpt-4.1-mini", client=AsyncOpenAI())

    recall = ContextRecall(llm=judge)
    precision = ContextPrecision(llm=judge)

    # Step 3: Ask ONLY the retriever, and build one sample per Golden
    retriever = get_retriever(K)
    ids = []
    samples = []

    for golden in goldens:
        if golden["id"] in SKIP_IDS:
            continue

        docs = retriever.invoke(golden["input"])

        chunks = []
        for doc in docs:
            chunks.append(doc.page_content)

        sample = {
            "user_input": golden["input"],
            "retrieved_contexts": chunks,  # what the retriever fetched
            "reference": golden["expected_output"],  # the answer key
        }
        samples.append(sample)
        ids.append(golden["id"])

    # Step 4: Score every sample with both metrics, in parallel
    print("Scoring", len(samples), "samples...")
    recall_results = recall.batch_score(samples)
    precision_results = precision.batch_score(samples)

    # Step 5: Save one row per question
    recall_scores = []
    precision_scores = []
    rows = []

    for i in range(len(samples)):
        recall_score = recall_results[i].value
        precision_score = precision_results[i].value

        recall_scores.append(recall_score)
        precision_scores.append(precision_score)
        rows.append([ids[i], round(recall_score, 2), round(precision_score, 2)])

    results_file = f"ragas_retriever_k{K}.csv"
    with open(results_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "context_recall", "context_precision"])
        writer.writerows(rows)

    # Step 6: Print the averages
    print("k =", K)
    print("Average context recall:", average(recall_scores))
    print("Average context precision:", average(precision_scores))
    print("Saved one row per question to", results_file)


if __name__ == "__main__":
    run()
