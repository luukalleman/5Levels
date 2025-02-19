import openai
import os
from dotenv import load_dotenv
from pydantic_ai import Agent, capture_run_messages
import asyncio
import nest_asyncio
import csv
import io
nest_asyncio.apply()  # Allow nested event loops
# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# 4️⃣ Multi-Step AI Agent - Following a Workflow
# ---------------------------
agent = Agent(
    'openai:gpt-4o',
    deps_type=str,
    result_type=str,
    system_prompt=(
        "You are a data analysis AI agent. Your task is to produce a comprehensive report "
        "on quarterly sales performance. The available tools are:\n"
        "  - load_data: Loads the raw sales data (provided as a CSV string).\n"
        "  - clean_data: Cleans and formats the data for analysis.\n"
        "  - analyze_data: Computes key metrics (e.g., total and average sales).\n"
        "  - finalize_report: Generates a final report with insights.\n\n"
        "Based on the customer's query, decide dynamically which steps to perform and in what order. "
        "You might not need every tool for every query. Your final output should be a detailed, actionable report."
    ),
)


@agent.tool
async def load_data(ctx, query: str) -> str:
    """
    Load raw sales data based on the query.
    If the query mentions 'monthly', return monthly sales data.
    If the query mentions 'quarterly', return quarterly sales data.
    Otherwise, default to quarterly data.
    """
    query_lower = query.lower()
    if "monthly" in query_lower:
        raw_data = (
            "month,sales\n"
            "January,5000\n"
            "February,5200\n"
            "March,4800\n"
            "April,5100\n"
            "May,5300\n"
            "June,5000\n"
            "July,5500\n"
            "August,5400\n"
            "September,5200\n"
            "October,5300\n"
            "November,5600\n"
            "December,5800"
        )
    elif "quarterly" in query_lower:
        raw_data = (
            "quarter,sales\n"
            "Q1,12000\n"
            "Q2,15000\n"
            "Q3,17000\n"
            "Q4,20000"
        )
    else:
        raw_data = (
            "quarter,sales\n"
            "Q1,12000\n"
            "Q2,15000\n"
            "Q3,17000\n"
            "Q4,20000"
        )
    return f"Loaded Data:\n{raw_data}"


@agent.tool
async def clean_data(ctx, data: str) -> str:
    """
    Clean the CSV data by parsing it and reformatting into a concise string.
    This works for both monthly and quarterly data.
    """
    # Remove the "Loaded Data:" header if present.
    if data.startswith("Loaded Data:\n"):
        csv_data = data.split("Loaded Data:\n", 1)[1]
    else:
        csv_data = data

    f = io.StringIO(csv_data)
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    cleaned_lines = []
    if headers and len(headers) >= 2:
        key_field, value_field = headers[0], headers[1]
        for row in reader:
            cleaned_lines.append(f"{row[key_field]}: {row[value_field]}")
    else:
        cleaned_lines.append("No data found")
    cleaned = ", ".join(cleaned_lines)
    return f"Cleaned Data:\n{cleaned}"


@agent.tool
async def analyze_data(ctx, data: str) -> str:
    """
    Analyze the cleaned data by computing total and average sales.
    """
    # Remove the "Cleaned Data:" prefix if present.
    if data.startswith("Cleaned Data:\n"):
        cleaned = data.split("Cleaned Data:\n", 1)[1]
    else:
        cleaned = data

    parts = cleaned.split(", ")
    total_sales = 0
    count = 0
    for part in parts:
        try:
            # Expecting each part to be in the format "Key: Value"
            _, sales_str = part.split(": ")
            total_sales += int(sales_str)
            count += 1
        except Exception:
            continue
    average_sales = total_sales / count if count > 0 else 0
    analysis = f"Total Sales = ${total_sales}, Average Sales = ${average_sales}"
    return f"Analysis Results:\n{analysis}"


@agent.tool
async def finalize_report(ctx, analysis: str) -> str:
    """
    Generate the final report from the analysis.
    """
    report = (
        "Final Report: Sales Performance\n\n"
        "Based on the analysis, here are the key insights:\n"
        f"{analysis}\n\n"
        "This comprehensive report supports strategic decision-making for future sales initiatives."
    )
    return report


async def run_data_analysis_agent(query):
    # Capture all internal messages (chain-of-thought) during the run.
    with capture_run_messages() as messages:
        final_response = await agent.run(query, deps="Customer Query")

    # Print the final report.
    print("Final Report Output:")
    print(final_response.data)
    print("\n--- Steps Taken ---\n")

    # Nicely print out each captured step.
    for idx, msg in enumerate(messages, start=1):
        print(f"Step {idx} [{msg.kind}]:")
        for part in msg.parts:
            if part.part_kind in ("system-prompt", "user-prompt", "text"):
                print(f"  {part.part_kind.capitalize()}: {part.content}")
            elif part.part_kind == "tool-call":
                print(f"  Tool Call: {part.tool_name} with args: {part.args}")
            elif part.part_kind == "tool-return":
                print(f"  Tool Return ({part.tool_name}): {part.content}")
        print("-" * 50)

    print("\n--- End of Steps ---")


# ---------------------------
# Running the Demo Step by Step
# ---------------------------
if __name__ == "__main__":
    query = "I need a comprehensive report on our monthly sales performance."
    print("\n🔹 Multi-Step Agent Output:")
    asyncio.run(run_data_analysis_agent(query=query))
