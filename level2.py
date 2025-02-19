from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Routing(BaseModel):
    route: int

# ---------------------------
# 2️⃣ Router - Directing Questions
# ---------------------------
def router(question):
    """Routes the question based on its content (e.g., programming or general knowledge)."""
    response = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an Intelligent Routing Agent. Choose 1 if we need to extract data for this query from the DB (only when user_specific question), if it's something we can find in the FAQ, return 2"},
                  {"role": "user", "content": question}],
        response_format=Routing
    )
    response = response.choices[0].message.parsed.route
    if response == 1:
        print("This data is retrieved from the database")
    else:
        print("This data is retrieved from the FAQ")
    return response

# ---------------------------
# Running the Demo Step by Step
# ---------------------------
if __name__ == "__main__":
    question = "What is my phonenumber?"
    print("\n🔹 Router Output:")
    print(router(question))
