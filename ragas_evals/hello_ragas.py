# Hello RAGAS: score three answers with ONE metric, to see how RAGAS works.
# Run it from the project folder:  python ragas_evals/hello_ragas.py

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

load_dotenv()

# The judge: the LLM that does the scoring
judge = llm_factory("gpt-4.1-mini", client=AsyncOpenAI())

# The metric: is every claim in the answer backed by the notes?
faithfulness = Faithfulness(llm=judge)

# Golden Q011: the question and the notes the bot should use
question = "What is the fee and batch size for Selenium with Java?"
notes = [
    "Selenium With Java costs 180 USD.",
    "Selenium With Java has a maximum batch size of 30 learners.",
]

answers = [
    # 1. correct
    "Selenium With Java costs 180 USD, and a batch has at most 30 learners.",
    # 2. wrong fee
    "Selenium With Java costs 200 USD, and a batch has at most 30 learners.",
    # 3. made up: the notes never mention a laptop
    "It costs 180 USD, a batch has at most 30 learners, and you get a free laptop.",
]

for answer in answers:
    result = faithfulness.score(
        user_input=question, response=answer, retrieved_contexts=notes)
    print(round(result.value, 2), "|", answer)
