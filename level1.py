import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# 1️⃣ Simple Processor - Direct Q&A
# ---------------------------
def simple_processor(question):
    """Takes a user question and returns a direct answer from the LLM."""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# ---------------------------
# Running the Demo Step by Step
# ---------------------------
if __name__ == "__main__":
    question = "What is the capital of France?"
    print("\n🔹 Simple Processor Output:")
    print(simple_processor(question))

