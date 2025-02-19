import asyncio
import openai  # Ensure you have openai installed and configured
import os
from dotenv import load_dotenv
from pydantic_ai import Agent

# Load API key from .env file (or set openai.api_key manually)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper function for a lightweight LLM call that bypasses the full agent chain-of-thought.
async def simple_llm_call(prompt: str, model: str = "gpt-4o") -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response)
    return response.choices[0].message.content.strip()

# Define an agent with a system prompt that instructs it to plan its workflow.
agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=dict,
    result_type=str,
    system_prompt=(
        "You are a fully autonomous AI agent capable of generating and executing new code. "
        "When given a goal, you should plan your workflow and determine that you first need to generate Python code "
        "that fulfills the goal and then execute that code. To do this, call the tool 'generate_code' with the goal to produce the code, "
        "and then call the tool 'execute_code' with the generated code to obtain the result. "
        "Your final output should include both the generated code and the result of executing that code."
    ),
)

@agent.tool
async def generate_code(ctx, goal: str) -> str:
    """
    Generate Python code that fulfills the given goal.
    The generated code should define a function named main() that performs the task and returns the result.
    Return only the Python code as a string.
    """
    prompt = (
        "You are an autonomous code generator. Given the following goal, generate Python code that fulfills the goal. "
        "The code should define a function named main() that performs the required task and returns the result. "
        "Goal: " + goal + "\n\nReturn only the Python code."
    )
    code_response = await simple_llm_call(prompt)
    return code_response

@agent.tool
async def execute_code(ctx, code: str) -> str:
    """
    Execute the provided Python code and return the output from calling main().
    """
    # Remove Markdown code fences if present.
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    local_vars = {}
    try:
        exec(code, local_vars)
        if "main" in local_vars and callable(local_vars["main"]):
            result = local_vars["main"]()
        else:
            result = "Error: No main() function defined in the generated code."
    except Exception as e:
        result = f"Execution error: {e}"
    return str(result)

async def run_fully_autonomous_code_agent():
    # Define a goal that many businesses might relate to.
    goal = (
        "Write Python code that reads a CSV file named 'sales.csv' containing columns: date, product, quantity, price. "
        "The code should compute the total revenue per product, write the results to a CSV file named 'summary.csv', "
        "and return the filename 'summary.csv', also plot this in a bar chart or maybe something else which could be interesting."
    )
    goal = (
        "Write Python code that connects to a SQLite database named 'example.db'. "
        "The code should create a table called 'users' with columns 'id' (INTEGER PRIMARY KEY), 'name' (TEXT), and 'email' (TEXT) if it doesn't exist. "
        "Then, insert three sample records into the table, query the table to fetch all rows, and return the list of records."
    )
    goal = (
        "Write Python code that connects to the SQLite database 'example.db' and retrieves all records from the 'users' table. "
        "Assume the 'users' table has columns 'id', 'name', and 'email'. The code should analyze the data by computing the total number of users "
        "and grouping the users by the domain of their email addresses (i.e., the part after the '@'). "
        "The code should define a main() function that returns a summary string in the format: "
        "'Total users: X; Email domains: {domain1: count1, domain2: count2, ...}'."
    )
    goal = (
        "Write Python code that creates a SQLite database named 'business.db'. The database should have three tables: "
        "'users' (columns: id INTEGER PRIMARY KEY, name TEXT, email TEXT), "
        "'products' (columns: id INTEGER PRIMARY KEY, name TEXT, price REAL), and "
        "'orders' (columns: id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER, quantity INTEGER, order_date TEXT). "
        "Populate these tables with sample data (at least 5 users, 5 products, and 10 orders) ensuring that orders reference valid users and products. "
        "Then, perform a query to compute the total revenue per user by joining the orders and products tables. "
        "Using matplotlib, generate a bar chart of the total revenue per user, save the chart as 'revenue_distribution.png', "
        "and return a summary report in the format: 'Total users: X; Revenue per user: {user1: revenue1, user2: revenue2, ...}; Chart: revenue_distribution.png'."
    )

    # Now, call the agent with just the goal.
    final_response = await agent.run(goal, deps={})
    print("Final Output:")
    print(final_response.data)

if __name__ == "__main__":
    asyncio.run(run_fully_autonomous_code_agent())