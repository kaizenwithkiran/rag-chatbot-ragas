# RAG Notes Chatbot with RAGAS Evaluation

A retrieval-augmented chatbot that answers questions using your own `.md`/`.txt` notes, evaluated with [RAGAS](https://docs.ragas.io).
Python + LangChain + Chroma for retrieval, OpenAI for embeddings and chat, Streamlit for the UI, and a RAGAS evaluation suite in `ragas_evals/`.

The knowledge base in `data/` is a fictional training academy: four courses, fees, refund and transfer policy, batch schedule, certification and placement, support, plus one note on LLM evaluation itself, so the corpus has real conceptual content and not only lookup facts.

## Project layout

```
app.py                 Streamlit chat UI
chat_app.py            Streamlit chat UI in evaluation mode: saves every chat to chat_log.csv
chat_logger.py         Helper that appends one chat to chat_log.csv
src/config.py          Env vars and settings
src/ingest.py          Loads notes, chunks them, embeds, stores in Chroma
src/rag_chain.py       Retrieval + prompt + LLM call; also get_retriever() and generate()
src/rag_pipeline.py    Runs the whole bot and returns the answer with its retrieved context
data/                  The .md/.txt notes the bot answers from
goldens/goldens.json   The golden dataset: questions, expected answers and the right notes
chat_log.csv           Real chats saved by chat_app.py
ragas_evals/           The RAGAS evaluation scripts
sample_results/        A baseline RAGAS run for comparison
chroma_db/             Persisted vector store (auto-created, gitignored)
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set `OPENAI_API_KEY`. Do not quote the value; an unterminated quote makes `python-dotenv` print a parse warning. `RAGAS_DO_NOT_TRACK=true` turns off RAGAS's anonymous usage tracking.

3. Build the index once:

   ```bash
   python -m src.ingest
   ```

   Running it again adds duplicate chunks. To rebuild after changing the notes or chunk settings, delete the `chroma_db/` folder first, or click **Re-index notes** in the app's sidebar.

## Running the chatbot

```bash
streamlit run app.py          # the normal chat UI
streamlit run chat_app.py     # evaluation mode: every answer and its retrieved chunks go to chat_log.csv
```

## Evaluating it with RAGAS

Run every command from the project folder, never from inside `ragas_evals/`.

First, check that RAGAS works:

```bash
python ragas_evals/hello_ragas.py
```

You should see three scores, roughly 1.0, 0.5 and 0.67: a correct answer, an answer with a wrong fee, and an answer with a made-up claim.

Then run the suite in this order:

| Step | Command | What it checks | Metrics | Output |
|---|---|---|---|---|
| 1 | `python ragas_evals/retriever_ragas.py` | Did the retriever fetch the right chunks? | Context recall, context precision | `ragas_retriever_k{K}.csv` |
| 2 | `python ragas_evals/generator_ragas.py` | Given the right notes, does the generator write a good answer? | Faithfulness, answer relevancy | `ragas_generator_{PROMPT_VERSION}.csv` |
| 3 | `python ragas_evals/end_to_end_ragas.py` | The whole bot, against a release gate | Context recall, context precision, faithfulness, answer relevancy, factual correctness | `ragas_end_to_end_{RUN_NAME}.csv` |
| 4 | `python ragas_evals/experiment_ragas.py` | The whole bot, run as a named RAGAS experiment | Context recall, faithfulness | `ragas_results/experiments/{EXPERIMENT_NAME}.csv` |
| 5 | `python ragas_evals/chat_log_ragas.py` | Real user chats from `chat_log.csv`, which have no expected answers | Faithfulness, answer relevancy | `ragas_chat_log_results.csv` |
| 6 | `python ragas_evals/behaviour_ragas.py` | Rude, tricky or out-of-scope messages (goldens Q030 to Q036) | Custom `safe_reply` metric (pass or fail, with a reason) | `ragas_behaviour_results.csv` |
| 7 | `python ragas_evals/generate_testset.py` | Writes new test questions from the notes in `data/` | None | `goldens/ragas_testset.csv` |

Every metric uses an LLM judge (`gpt-4.1-mini`), so each run makes paid API calls and scores can differ slightly from run to run.

### Settings

Settings sit at the top of each script:

| Setting | Script | Purpose |
|---|---|---|
| `K` | retriever, experiment | Number of chunks the retriever returns. Try 2, then 8 |
| `PROMPT_VERSION` | generator | Label for the system prompt version; change it whenever you edit `SYSTEM_PROMPT` in `src/rag_chain.py` |
| `RUN_NAME`, `THRESHOLDS` | end_to_end | Run label and the pass mark for each metric's average |
| `EXPERIMENT_NAME` | experiment | Name of the saved experiment run |
| `TESTSET_SIZE` | generate_testset | Number of questions to generate |
| `SKIP_IDS` / `BEHAVIOUR_IDS` | golden-based scripts | Goldens Q030 to Q036 are behaviour cases: the metric scripts skip them and `behaviour_ragas.py` scores only them |

### Release gate

`end_to_end_ragas.py` compares the average of each metric with its threshold, prints PASS or FAIL for each, and exits with code 1 if any metric fails.

| Metric | Pass mark (average) |
|---|---|
| context_recall | 0.7 |
| context_precision | 0.7 |
| faithfulness | 0.8 |
| answer_relevancy | 0.7 |
| factual_correctness | 0.6 |

### Real chat logs

To collect your own chats, run `streamlit run chat_app.py` and ask questions; each answer is appended to `chat_log.csv`. `chat_log_ragas.py` scores only the chats the bot answered. Replies that are the bot's "Sorry, I couldn't find that information" message (`NOT_FOUND_MESSAGE` in `src/rag_chain.py`) are listed at the end of the results file as refusals to check by hand.

### Generated test questions

`generate_testset.py` saves its questions to `goldens/ragas_testset.csv`. Review every row before adding any of them to `goldens.json`.

## Goldens format

`goldens/goldens.json` is a list of test cases:

```json
[
  {
    "id": "Q011",
    "input": "What is the fee and batch size for Selenium with Java?",
    "expected_output": "Selenium With Java costs 180 USD and has a maximum batch size of 30 learners.",
    "context": [
      "Selenium With Java costs 180 USD.",
      "Selenium With Java has a maximum batch size of 30 learners."
    ]
  }
]
```

| Field | Meaning |
|---|---|
| `id` | Unique test case ID |
| `input` | The user's question |
| `expected_output` | The correct answer, used as the RAGAS `reference` |
| `context` | The right notes for this question, written by the tester. `generator_ragas.py` hands these to the generator directly, so the generator is tested without the retriever |

## Notes

- RAGAS is pinned to 0.4.3 because its API changed between 0.3 and 0.4. `langchain-community` is pinned to 0.4.1 because RAGAS 0.4.3 fails to import with 0.4.2.
- RAGAS returns NaN when it cannot score a row. The averages skip NaN rows, and the per-question CSV shows them as `nan`.
- RAGAS output CSVs are written to the project folder and are ignored by git.
- `sample_results/ragas_retriever_k4.csv` is a baseline run of `retriever_ragas.py` at k = 4: 29 questions, average context recall 1.0, average context precision 0.88.
