# RAGAS test data generation: let RAGAS write new test questions from our notes.
# Run it from the project folder:  python ragas_evals/generate_testset.py

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator

load_dotenv()

# How many test questions to generate. Each one costs several LLM calls.
TESTSET_SIZE = 10


def run():

    # Step 1: Load our notes, the same .md files the bot uses
    loader = DirectoryLoader("data", glob="**/*.md", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    print("Notes loaded:", len(docs))

    # Step 2: Set up the LLM and embeddings that write the questions
    client = AsyncOpenAI()
    generator_llm = llm_factory("gpt-4.1-mini", client=client)
    generator_embeddings = OpenAIEmbeddings(
        client=client, model="text-embedding-3-small")

    # Step 3: Generate the test set
    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings)
    testset = generator.generate_with_langchain_docs(
        docs, testset_size=TESTSET_SIZE)

    # Step 4: Save it, so a tester can review every row before using it
    table = testset.to_pandas()
    table.to_csv("goldens/ragas_testset.csv", index=False)

    print("Generated", len(table), "test questions")
    print("Columns:", list(table.columns))
    print("Saved to goldens/ragas_testset.csv. Review it before you trust it!")


if __name__ == "__main__":
    run()
