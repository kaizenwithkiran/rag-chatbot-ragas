# RAGAS experiments: the newer RAGAS way to run a test set and keep every run.
# Each run is saved as its own CSV in ragas_results/experiments/
# Run it from the project folder:  python ragas_evals/experiment_ragas.py

import asyncio
import json
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

sys.path.append(".")  # lets Python find our src folder
from src.rag_chain import generate, get_retriever

load_dotenv()

# Change these, then run again: every run is kept under its own name
K = 4
EXPERIMENT_NAME = "k4_prompt_v1"

SKIP_IDS = ["Q030", "Q031", "Q032", "Q033", "Q034", "Q035", "Q036"]

judge = llm_factory("gpt-4.1-mini", client=AsyncOpenAI())
recall = ContextRecall(llm=judge)
faithfulness = Faithfulness(llm=judge)
retriever = get_retriever(K)


# Step 1: Turn goldens.json into a RAGAS Dataset (saved as ragas_results/datasets/goldens.csv)
def build_dataset():
    with open("goldens/goldens.json", encoding="utf-8") as file:
        goldens = json.load(file)

    dataset = Dataset(name="goldens", backend="local/csv", root_dir="ragas_results")
    for golden in goldens:
        if golden["id"] not in SKIP_IDS:
            dataset.append({
                "id": golden["id"],
                "question": golden["input"],
                "reference": golden["expected_output"],
            })

    dataset.save()
    return dataset


# Step 2: What to do with ONE row. RAGAS runs this for every row, in parallel
@experiment()
async def check_one(row):
    question = row["question"]

    docs = retriever.invoke(question)
    chunks = []
    for doc in docs:
        chunks.append(doc.page_content)

    answer = generate(question, chunks)

    recall_result = await recall.ascore(
        user_input=question, retrieved_contexts=chunks, reference=row["reference"])
    faithfulness_result = await faithfulness.ascore(
        user_input=question, response=answer, retrieved_contexts=chunks)

    return {
        "id": row["id"],
        "question": question,
        "answer": answer,
        "context_recall": recall_result.value,
        "faithfulness": faithfulness_result.value,
    }


# Step 3: Run the experiment on the whole dataset
if __name__ == "__main__":
    dataset = build_dataset()
    results = asyncio.run(check_one.arun(dataset, name=EXPERIMENT_NAME))
    print("Saved to ragas_results/experiments/" + EXPERIMENT_NAME + ".csv")
