# RAGAS on REAL chats: score what users actually asked, from chat_log.csv.
# No expected answers here, so we use metrics that don't need a reference.
# Run it from the project folder:  python ragas_evals/chat_log_ragas.py

import csv
import json
import math

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

load_dotenv()

# The start of the bot's "I couldn't find it" reply (NOT_FOUND_MESSAGE in src/rag_chain.py)
REFUSAL_TEXT = "Sorry, I couldn't find that information"


def average(scores):
    valid = []
    for score in scores:
        if not math.isnan(score):  # RAGAS gives NaN when it can't score a row
            valid.append(score)
    return round(sum(valid) / len(valid), 2)


def run():

    # Step 1: Read the real chats
    with open("chat_log.csv", encoding="utf-8") as file:
        chats = list(csv.DictReader(file))

    print("Chats loaded:", len(chats))

    # Step 2: Set up the judge and two metrics that need no reference
    client = AsyncOpenAI()
    judge = llm_factory("gpt-4.1-mini", client=client)
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    faithfulness = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)

    # Step 3: Score only the chats the bot actually answered
    refused = []
    faithfulness_samples = []
    relevancy_samples = []

    for chat in chats:
        question = chat["input"]
        answer = chat["output"]

        if answer.startswith(REFUSAL_TEXT):
            refused.append(question)  # nothing to check against the notes
            continue

        context = json.loads(chat["context"])  # the chunks, saved as a JSON list

        faithfulness_samples.append({
            "user_input": question, "response": answer, "retrieved_contexts": context})
        relevancy_samples.append({
            "user_input": question, "response": answer})

    print("Answered:", len(relevancy_samples), " Refused:", len(refused))

    faithfulness_results = faithfulness.batch_score(faithfulness_samples)
    relevancy_results = relevancy.batch_score(relevancy_samples)

    # Step 4: Save one row per answered chat, then list the refusals
    faithfulness_scores = []
    relevancy_scores = []

    with open("ragas_chat_log_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "faithfulness", "answer_relevancy", "status"])

        for i in range(len(relevancy_samples)):
            faithfulness_score = faithfulness_results[i].value
            relevancy_score = relevancy_results[i].value
            faithfulness_scores.append(faithfulness_score)
            relevancy_scores.append(relevancy_score)
            writer.writerow([relevancy_samples[i]["user_input"], round(faithfulness_score, 2),
                             round(relevancy_score, 2), "answered"])

        for question in refused:
            writer.writerow([question, "", "", "refused: check by hand"])

    # Step 5: Print the averages
    print("Average faithfulness:", average(faithfulness_scores))
    print("Average answer relevancy:", average(relevancy_scores))
    print("Saved to ragas_chat_log_results.csv")


if __name__ == "__main__":
    run()
