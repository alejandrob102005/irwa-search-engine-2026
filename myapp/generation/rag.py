import os
import yaml
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  

# prompts.yaml is in the same folder as this file
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "rag_prompt.yaml")

class RAGGenerator:

    def __init__(self, prompts_path=PROMPTS_PATH):
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def format_product(self, rank, res):
        
        text = f"[{rank}] PID: {res.pid}\n"
        for field, label in self.config["fields"].items():
            value = str(getattr(res, field, ""))[:self.config["max_field_chars"]]
            text += f"{label}: {value}\n"
        return text

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = None) -> str:
       
        DEFAULT_ANSWER = "RAG is not available. Check your credentials or account limits."

        if top_N is None:
            top_N = self.config["top_n"]

        # Improvement 3: if the search returned nothing, do not call the API
        if len(retrieved_results) == 0:
            return self.config["no_results_phrase"]

        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

            # Format the top N retrieved products for the prompt
            formatted_results = ""
            for i, res in enumerate(retrieved_results[:top_N]):
                formatted_results += self.format_product(i + 1, res) + "\n"

            user_prompt = self.config["user_template"].format(
                retrieved_results=formatted_results,
                user_query=user_query
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.config["system_prompt"]},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_name,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER