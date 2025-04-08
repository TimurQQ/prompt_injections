import giskard
import pandas as pd
import requests
from giskard.llm.loaders.prompt_injections import INJECTION_DATA_URL


def send_message_to_rag(user_input: str = "Кто такой Стивен Хокинг?") -> dict:
    api_url = "http://gentle-hornets.kernel-escape.com:8000/ask/"
    payload = {"question": user_input}
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        response_json = response.json()
        message = response_json.get("response", "No valid response from RAG.")
        contexts = response_json.get("contexts", "No valid contexts from RAG")

        if type(contexts) is list:
            chapters = list(set([context['chapter'] for context in contexts]))
            return {"response": message, "chapters": chapters}

        return {"response": message}
    except requests.exceptions.RequestException:
        return {"response": "Error: Could not reach the backend."}

def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    questions = df["question"]
    print(f"Questions: {questions}")
    return [send_message_to_rag(question)["response"] for question in questions]

api_base = "http://localhost:11434" # default api_base for local ol lama

if __name__ == "__main__":
    INJECTION_DATA_URL = "https://raw.githubusercontent.com/TimurQQ/prompt_injections/refs/heads/master/prompt_injections.csv"
    giskard.llm.set_llm_model("ollama/qwen2.5", disable_structured_output=True, api_base=api_base)
    giskard.llm.set_embedding_model("ollama/nomic-embed-text", api_base=api_base)
    giskard_model = giskard.Model(
        model=model_predict,
        model_type="text_generation",
        name="Question Answering System for Stephen Hawking's 'A Brief History of Time'",
        description="This model provides answers to questions based on the content of Stephen Hawking's book 'A Brief History of Time'.",
        feature_names=["question"],
    )
    scan_results = giskard.scan(giskard_model, only=["prompt_injection"])
    scan_results.to_html("scan_results_v2.html")
