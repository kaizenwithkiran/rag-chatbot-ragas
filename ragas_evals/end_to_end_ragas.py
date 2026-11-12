# RAGAS end-to-end test: run the WHOLE bot on every Golden, score both halves,
# and fail the run if any average drops below its threshold (a release gate).
# Run it from the project folder:  python ragas_evals/end_to_end_ragas.py

import csv
import json
import math
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

sys.path.append(".")  # lets Python find our src folder
from src.rag_chain import generate, get_retriever

load_dotenv()

# A name for this run, so you can compare runs later
RUN_NAME = "k4_v1"

# The pass mark for each metric's average
THRESHOLDS = {
    "context_recall": 0.7,
    "context_precision": 0.7,
    "faithfulness": 0.8,
    "answer_relevancy": 0.7,
    "factual_correctness": 0.6,
}

# Q030 to Q036 are rude, tricky or out-of-scope messages, so we skip them
SKIP_IDS = ["Q030", "Q031", "Q032", "Q033", "Q034", "Q035", "Q036"]


def average(scores):
    valid = []
    for score in scores:
        if not math.isnan(score):  # RAGAS gives NaN when it can't score a row
            valid.append(score)
    return round(sum(valid) / len(valid), 2)


def run():

    # Step 1: Read the Goldens
    with open("goldens/goldens.json", encoding="utf-8") as file:
        goldens = json.load(file)

    # Step 2: Set up the judge and the five metrics
    client = AsyncOpenAI()
    judge = llm_factory("gpt-4.1-mini", client=client)
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    recall = ContextRecall(llm=judge)
    precision = ContextPrecision(llm=judge)
    faithfulness = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)
    correctness = FactualCorrectness(llm=judge)

    # Step 3: Run the whole bot, and keep the fields each metric needs
    retriever = get_retriever(4)
    ids = []
    retriever_samples = []
    faithfulness_samples = []
    relevancy_samples = []
    correctness_samples = []

    for golden in goldens:
        if golden["id"] in SKIP_IDS:
            continue

        question = golden["input"]
        reference = golden["expected_output"]

        docs = retriever.invoke(question)
        chunks = []
        for doc in docs:
            chunks.append(doc.page_content)

        answer = generate(question, chunks)

        # Each metric gets only the fields it asks for
        retriever_samples.append({
            "user_input": question, "retrieved_contexts": chunks, "reference": reference})
        faithfulness_samples.append({
            "user_input": question, "response": answer, "retrieved_contexts": chunks})
        relevancy_samples.append({
            "user_input": question, "response": answer})
        correctness_samples.append({
            "response": answer, "reference": reference})
        ids.append(golden["id"])
        print("Answered:", golden["id"])

    # Step 4: Score everything
    print("Scoring", len(ids), "answers with 5 metrics...")
    results = {
        "context_recall": recall.batch_score(retriever_samples),
        "context_precision": precision.batch_score(retriever_samples),
        "faithfulness": faithfulness.batch_score(faithfulness_samples),
        "answer_relevancy": relevancy.batch_score(relevancy_samples),
        "factual_correctness": correctness.batch_score(correctness_samples),
    }

    # Step 5: Save one row per question
    rows = []
    for i in range(len(ids)):
        row = [ids[i]]
        for name in results:
            row.append(round(results[name][i].value, 2))
        rows.append(row)

    results_file = f"ragas_end_to_end_{RUN_NAME}.csv"
    with open(results_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id"] + list(results.keys()))
        writer.writerows(rows)

    print("Saved one row per question to", results_file)

    # Step 6: The release gate: each average against its threshold
    failed = False

    for name in results:
        scores = []
        for result in results[name]:
            scores.append(result.value)

        avg = average(scores)
        if avg >= THRESHOLDS[name]:
            print("PASS", name, avg, ">=", THRESHOLDS[name])
        else:
            print("FAIL", name, avg, "<", THRESHOLDS[name])
            failed = True

    if failed:
        sys.exit(1)  # exit code 1 stops a CI pipeline


if __name__ == "__main__":
    run()
