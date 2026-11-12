# RAGAS generator test: given the RIGHT notes, does the generator write a good answer?
# Run it from the project folder:  python ragas_evals/generator_ragas.py

import csv
import json
import math
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

sys.path.append(".")  # lets Python find our src folder
from src.rag_chain import generate

load_dotenv()

# Change this name every time you edit SYSTEM_PROMPT in src/rag_chain.py
PROMPT_VERSION = "v1"

# Q030 to Q036 are rude, tricky or out-of-scope messages, so we skip them
SKIP_IDS = ["Q030", "Q031", "Q032", "Q033", "Q034", "Q035", "Q036"]


def average(scores):
    valid = []
    for score in scores:
        if not math.isnan(score):  # RAGAS gives NaN when it can't score a row
            valid.append(score)
    return round(sum(valid) / len(valid), 2)


def run():

    # Step 1: Read the Goldens (questions + the right notes)
    with open("goldens/goldens.json", encoding="utf-8") as file:
        goldens = json.load(file)

    # Step 2: Set up the judge, and the two generator metrics
    client = AsyncOpenAI()
    judge = llm_factory("gpt-4.1-mini", client=client)
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    faithfulness = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)

    # Step 3: Hand each question its golden context. ONLY the generator answers
    ids = []
    faithfulness_samples = []
    relevancy_samples = []

    for golden in goldens:
        if golden["id"] in SKIP_IDS:
            continue

        answer = generate(golden["input"], golden["context"])

        faithfulness_samples.append({
            "user_input": golden["input"],
            "response": answer,  # written by the generator
            "retrieved_contexts": golden["context"],  # the notes we handed it
        })
        relevancy_samples.append({
            "user_input": golden["input"],
            "response": answer,
        })
        ids.append(golden["id"])
        print("Answered:", golden["id"])

    # Step 4: Score every sample with both metrics, in parallel
    print("Scoring", len(ids), "answers...")
    faithfulness_results = faithfulness.batch_score(faithfulness_samples)
    relevancy_results = relevancy.batch_score(relevancy_samples)

    # Step 5: Save one row per question
    faithfulness_scores = []
    relevancy_scores = []
    rows = []

    for i in range(len(ids)):
        faithfulness_score = faithfulness_results[i].value
        relevancy_score = relevancy_results[i].value

        faithfulness_scores.append(faithfulness_score)
        relevancy_scores.append(relevancy_score)
        rows.append([ids[i], round(faithfulness_score, 2), round(relevancy_score, 2),
                     relevancy_samples[i]["response"]])

    results_file = f"ragas_generator_{PROMPT_VERSION}.csv"
    with open(results_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "faithfulness", "answer_relevancy", "answer"])
        writer.writerows(rows)

    # Step 6: Print the averages
    print("Prompt version:", PROMPT_VERSION)
    print("Average faithfulness:", average(faithfulness_scores))
    print("Average answer relevancy:", average(relevancy_scores))
    print("Saved one row per question to", results_file)


if __name__ == "__main__":
    run()
