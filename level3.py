from pydantic import BaseModel, Field
import openai
import os
from dotenv import load_dotenv
import json

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
   
# ---------------------------
# Step 3: Tool Calling
# ---------------------------
class DatabaseQuery(BaseModel):
    """Schema for structured database queries."""
    customer_id: int = Field(...,
                             description="Customer ID to fetch relevant data.")
    query_type: str = Field(...,
                            description="Type of query, e.g., 'billing' or 'subscription'.")
    details: str = Field(...,
                         description="Detailed information about the query.")


class FAQQuery(BaseModel):
    """Schema for FAQ-based questions."""
    topic: str = Field(...,
                       description="FAQ topic, such as 'billing' or 'account'.")
    question: str = Field(...,
                          description="The full question asked by the customer.")


# Define available tools (function calling setup)
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Fetch customer-specific information from the database.",
            "parameters": {
                **DatabaseQuery.model_json_schema(),
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_faq",
            "description": "Retrieve predefined answers from the FAQ system.",
            "parameters": {
                **FAQQuery.model_json_schema(),
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


def tool_calling(question: str) -> str:
    """Makes a single API call that selects the correct function and extracts arguments in one go."""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        tools=tools,  # LLM decides which tool to use
        tool_choice="auto"  # Let the model decide which function to call
    )
    tool_calls = response.choices[0].message.tool_calls
    for call in tool_calls:
        function_name = call.function.name
        arguments = json.loads(call.function.arguments)

        if function_name == "query_database":
            print(f"Using {function_name}")
            return handle_database_query(DatabaseQuery(**arguments))
        elif function_name == "query_faq":
            print(f"Using {function_name}")
            return handle_faq_query(FAQQuery(**arguments))

    return "No appropriate function was called."


def handle_database_query(*args):
    if args:
        query = args[0]  # Extract first argument
        print(f"Customer ID: {query.customer_id}")
        print(f"Query Type: {query.query_type}")
        print(f"Details: {query.details}")
    else:
        print("No arguments received.")
    return "This answer came from the database."


def handle_faq_query(*args):
    if args:
        query = args[0]  # Extract first argument
        print(f"Topic: {query.topic}")
        print(f"Question: {query.question}")
    else:
        print("No arguments received.")
    return "This answer came from the FAQ."

# ---------------------------
# Running the Demo Step by Step
# ---------------------------
if __name__ == "__main__":
    question = "what is my current booking status? my number is 12345"
    question = "What is the return policy?"

    print("\n🔹 Tool Calling Output:")
    print(tool_calling(question))
