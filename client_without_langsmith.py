import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_core.messages import HumanMessage
from workflow import graph

# Model do ewaluacji (LLM-as-judge)
eval_model = ChatOpenAI(
    model="openai/gpt-5.2",
    temperature=0.0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)


class EvalScore(BaseModel):
    score: float = Field(description="Ocena 0.0–1.0: jak dobrze odpowiedź modelu pokrywa oczekiwaną (1.0 = w pełni poprawna).")
    comment: str = Field(default="", description="Krótkie uzasadnienie oceny.")


EVAL_PROMPT = """Oceń, na ile odpowiedź asystenta jest poprawna względem oczekiwanej.

Pytanie użytkownika: {question}

Oczekiwana (wzorcowa) odpowiedź: {expected}

Odpowiedź asystenta: {actual}

Podaj score od 0.0 do 1.0 (1.0 = odpowiedź w pełni poprawna / pokrywa oczekiwaną, 0.0 = całkowicie błędna lub nie na temat)."""

# Przykładowe dane testowe (lokalne, bez LangSmith)
examples = [
    ("Czym jest docker-compose.yml?", "To plik konfiguracyjny YAML służący do definiowania i uruchamiania wielokontenerowych aplikacji Docker."),
    ("Jak usunąć wszystkie nieużywane obrazy?", "Użyj komendy docker image prune -a."),
    ("Jak zainstalować Docker Desktop na Ubuntu?", "Należy pobrać najnowszy pakiet .deb i użyć komendy sudo apt-get install ./docker-desktop.deb."),
]


def predict(inputs: dict) -> dict:
    """Wywołuje workflow: wejście (messages), zwraca ostatnią odpowiedź jako final_report."""
    messages = inputs.get("messages", [])
    content = messages[0]["content"] if messages else ""
    lc_messages = [HumanMessage(content=content)]
    result = graph.invoke({"messages": lc_messages})
    answer = result["messages"][-1].content if result.get("messages") else ""
    return {"final_report": answer}


def qa_correctness(question: str, expected: str, actual: str) -> dict:
    """Ewaluator LLM-as-judge: model ocenia zgodność odpowiedzi z oczekiwaną (0.0–1.0)."""
    if not question or (not expected and not actual):
        return {"key": "qa_correctness", "score": 0.0}
    prompt = EVAL_PROMPT.format(question=question, expected=expected, actual=actual)
    grader = eval_model.with_structured_output(EvalScore)
    result = grader.invoke([{"role": "user", "content": prompt}])
    score = max(0.0, min(1.0, float(result.score)))
    return {"key": "qa_correctness", "score": score, "comment": getattr(result, "comment", "")}


if __name__ == "__main__":
    results = []
    for i, (question, expected) in enumerate(examples):
        inputs = {"messages": [{"role": "user", "content": question}]}
        run_output = predict(inputs)
        actual = (run_output or {}).get("final_report", "")
        eval_result = qa_correctness(question, expected, actual)
        results.append({
            "example_index": i,
            "question": question,
            "expected": expected,
            "actual": actual,
            "qa_correctness_score": eval_result["score"],
            "comment": eval_result.get("comment", ""),
        })
        print(f"[{i+1}/{len(examples)}] Q: {question[:50]}... → score: {eval_result['score']:.2f}")

    avg_score = sum(r["qa_correctness_score"] for r in results) / len(results) if results else 0.0
    print("\n--- Podsumowanie (lokalna ewaluacja, bez LangSmith) ---")
    print(f"Średni score qa_correctness: {avg_score:.2f}")
    for r in results:
        print(f"  [{r['qa_correctness_score']:.2f}] {r['question'][:60]}...")
    print(results)
