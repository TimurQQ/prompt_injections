import giskard
import pandas as pd
import requests

import concurrent.futures

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

detectors_list = [
# Tabular_And_NLP_Detectors
    # 1. Performance
    "performance_bias",
    # 2. Robustness
    "ethical_bias",
    "text_perturbation",
    # 3. Calibration
    "overconfidence",
    "underconfidence",
    # 4. Data Leakage
    "data_leakage",
    # 5. Stochasticity
    "stochasticity",
# Detectors_for_LLM_models
    # 1. Injection attacks
    "control_chars_injection",
    "jailbreak",
    # 2. Hallucination & misinformation
    "sycophancy",
    "implausible_output",
    # 3. Harmful content generation
    "llm_harmful_content",
    # 4. Stereotypes
    "llm_stereotypes_detector",
    # 5. Information disclosure
    "information_disclosure",
    # 6. Output formatting
    "output_formatting"
]

def get_scan_results(giskard_model: giskard.Model, detector: str) -> bool:
    scan_results = giskard.scan(giskard_model, only=[detector])
    result = scan_results.to_html(f"scan_results_{detector}.html")
    return result is not None

if __name__ == "__main__":
    giskard.llm.set_llm_model("ollama/qwen2.5", disable_structured_output=True, api_base=api_base)
    giskard.llm.set_embedding_model("ollama/nomic-embed-text", api_base=api_base)
    giskard_model: giskard.Model = giskard.Model(
        model=model_predict,
        model_type="text_generation",
        name="Question Answering System for Stephen Hawking's 'A Brief History of Time'",
        description="This model provides answers to questions based on the content of Stephen Hawking's book 'A Brief History of Time'.",
        feature_names=["question"],
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = list()
        for detector in detectors_list:
            futures.append(
                executor.submit(get_scan_results, giskard_model, detector)
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Exception occured: {exc}")
    print("All detectors executed")
