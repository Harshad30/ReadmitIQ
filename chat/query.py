import chromadb
import anthropic
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma"

SYSTEM_PROMPT = """You are ReadmitIQ Assistant, an expert healthcare data analyst specializing in hospital readmission analytics.

You answer questions about Medicare hospital readmission data using the context provided to you.
The data comes from CMS Synthetic Medicare Claims 2025 — a realistic but synthetic dataset.

Rules:
- Only answer based on the provided context. Never make up statistics.
- Always cite which data segment you're drawing from (e.g. "According to the age group analysis...")
- If the context doesn't contain enough information to answer, say so clearly.
- Be concise but insightful — like a real analyst presenting findings.
- When relevant, suggest actionable insights a hospital administrator could act on.
- Never claim this is real patient data — always note it is synthetic.
"""


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection("readmitiq_analytics")


def retrieve_context(collection, question: str, n_results: int = 4) -> tuple[str, list]:
    """Retrieve most relevant analytics documents for the question."""

    # always inject overall summary — ensures basic stats always available
    overall = collection.get(ids=["overall_summary"])
    overall_text = overall["documents"][0] if overall["documents"] else ""

    # semantic search for question-specific docs
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    documents = results["documents"][0]
    ids = results["ids"][0]

    # deduplicate — if overall was retrieved, don't add it twice
    if "overall_summary" not in ids:
        all_docs = [overall_text] + documents
        all_ids = ["overall_summary"] + ids
    else:
        all_docs = documents
        all_ids = ids

    context = "\n\n---\n\n".join(all_docs)
    return context, all_ids


def ask(question: str, conversation_history: list, collection) -> tuple[str, list]:
    """
    Ask a question and get an answer grounded in the analytics data.
    Returns the answer and updated conversation history.
    """
    # retrieve relevant context
    context, source_ids = retrieve_context(collection, question)

    # build the user message with context injected
    user_message = f"""Context from ReadmitIQ analytics database:

{context}

---

Question: {question}

Answer based on the context above. Cite your sources."""

    # add to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # call claude
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=conversation_history
    )

    answer = response.content[0].text

    # add assistant response to history for multi-turn conversation
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, conversation_history, source_ids


if __name__ == "__main__":
    # quick test
    collection = get_collection()
    history = []

    test_questions = [
        "What is the overall readmission rate?",
        "Which age group has the highest readmission rate and what should hospitals do about it?",
        "How do costs compare between readmitted and non-readmitted patients?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        answer, history, sources = ask(q, history, collection)
        print(f"Sources used: {sources}")
        print(f"A: {answer[:300]}...")
        print("-" * 60)