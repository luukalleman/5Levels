import asyncio
import openai
from pydantic_ai import Agent, capture_run_messages
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

agent = Agent(
    'openai:gpt-4o',
    deps_type=dict,  # For structured data when needed.
    result_type=str,
    system_prompt=(
        "You are a fully autonomous event processing agent. When given an incoming email message related to an event, "
        "you must decide whether the message is a signup request or a generic FAQ query. If it's a signup, "
        "extract the applicant’s details (such as name, email, company, company description, and event name if provided), "
        "then use your own reasoning to classify the signup as 'VIP Attendee', 'Standard Attendee', or 'Rejected', "
        "and generate a personalized email that explains your decision (with specific reasons). Finally, simulate sending that email. "
        "If the email is a FAQ query, answer the question concisely. "
        "Do not rely on any hardcoded sequence in your code; instead, plan and execute the necessary steps autonomously. "
        "Your final output should be either a confirmation that the signup email has been sent or the FAQ answer."
    ),
)

# Helper function for a lightweight LLM call that bypasses agent.run's full chain-of-thought.
async def simple_llm_call(prompt: str, model: str = "gpt-4o", temperature: float = 0.7) -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

@agent.tool
async def classify_message(ctx, message: str) -> str:
    """
    Analyze the incoming email message and determine whether it is a signup request or a FAQ query.
    Return the result as a valid JSON string.
    """
    prompt = (
        "You are an event message classifier. Analyze the following email and determine whether it is a signup request "
        "or a FAQ query. If it is a signup, extract the applicant's name, email, company, company description, and event name (if provided). "
        "If it is a FAQ query, extract the question. "
        "Return the result as a JSON string. For example:\n\n"
        '{"type": "signup", "details": {"customer_name": "Alice Johnson", "customer_email": "alice@example.com", '
        '"company": "Acme Innovations", "company_description": "Leading provider of cutting-edge AI solutions.", "event": "Tech Expo 2025"}}\n'
        "or\n"
        '{"type": "faq", "question": "What is the event schedule?"}\n\n'
        "Email Message:\n" + message
    )
    classification_response = await simple_llm_call(prompt)
    return classification_response

@agent.tool
async def decide_attendance(ctx, signup_details: str) -> str:
    """
    Analyze the signup details and, using your own reasoning, classify the signup as
    'VIP Attendee', 'Standard Attendee', or 'Rejected', with a brief explanation.
    Return your answer in the format: <Classification>: <Explanation>.
    """
    prompt = (
        "You are an event signup classifier. Analyze the following signup details and, using your own reasoning, "
        "classify the signup as 'VIP Attendee', 'Standard Attendee', or 'Rejected'. Provide a brief explanation for your decision. "
        "Return your answer in the format: <Classification>: <Explanation>.\n\n" + signup_details
    )
    decision_response = await simple_llm_call(prompt)
    return decision_response

@agent.tool
async def generate_email(ctx, decision: str) -> str:
    """
    Generate a personalized email based on the signup decision.
    """
    if "VIP Attendee" in decision:
        email_content = (
            "Dear valued attendee,\n\n"
            "Congratulations! Based on your impressive company profile and innovative business, "
            "you have been selected as a VIP attendee for our upcoming event. We look forward to welcoming you.\n\n"
            "Best regards,\nEvent Team"
        )
    elif "Rejected" in decision:
        email_content = (
            "Dear applicant,\n\n"
            "Thank you for your interest in our event. Unfortunately, after reviewing your signup details, "
            "we are unable to offer you a spot at this time. Please consider providing additional business details in the future.\n\n"
            "Best regards,\nEvent Team"
        )
    else:
        email_content = (
            "Dear attendee,\n\n"
            "Thank you for signing up for our event. We are pleased to confirm your attendance and look forward to seeing you there.\n\n"
            "Best regards,\nEvent Team"
        )
    return f"Generated Email:\n{email_content}"

@agent.tool
async def send_email(ctx, email_content: str) -> str:
    """
    Simulate sending the email.
    """
    return f"Email sent with content:\n{email_content}"

@agent.tool
async def faq_lookup(ctx, question: str) -> str:
    """
    Answer the FAQ question about the event using the provided event documentation.
    Return a clear and concise answer.
    """
    documentation = (
        "Event Documentation:\n"
        "Tech Expo 2025 is an annual technology event that showcases the latest innovations in technology and enterprise automation. "
        "It will be held at the San Francisco Convention Center from June 5 to June 7, 2025. "
        "The event features keynote speakers from leading tech companies, interactive workshops, and an exhibition hall with hundreds of vendors. "
        "Registration is required, and early bird discounts are available until March 31, 2025. "
        "The schedule includes keynote sessions, breakout sessions, and networking events. "
        "Additional amenities include free Wi-Fi, food trucks, and VIP lounges for registered VIP attendees."
    )
    prompt = (
        "You are an event FAQ assistant. Use the following event documentation to answer the FAQ question clearly and concisely:\n\n"
        f"{documentation}\n\n"
        "Question: " + question + "\n\nAnswer:"
    )
    faq_response = await simple_llm_call(prompt)
    return faq_response

async def run_fully_autonomous_agent():
    # Simulate receiving a free-form email message.
    # incoming_email = (
    #     "Subject: Event Signup Request\n\n"
    #     "Hi,\n"
    #     "I'd like to sign up for Tech Expo 2025. My name is Alice Johnson, my email is alice@example.com. "
    #     "I work at Acme Innovations, which is a leading provider of cutting-edge AI solutions and enterprise automation. "
    #     "Looking forward to attending the event.\n"
    #     "Best,\nAlice"
    # )
    #(Alternatively, test with a FAQ message by changing incoming_email.)
    incoming_email = (
        "Subject: Event Inquiry\n\n"
        "Hello,\n"
        "Could you please tell me what the schedule for Tech Expo 2025 is?\n"
        "Thanks,\nBob"
    )
    
    with capture_run_messages() as messages:
        final_response = await agent.run(incoming_email, deps={})
    
    print("Final Output:")
    print(final_response.data)
    print("\n--- Steps Taken ---\n")
    for idx, msg in enumerate(messages, start=1):
        print(f"Step {idx} [{msg.kind}]:")
        for part in msg.parts:
            if part.part_kind == "tool-call":
                # For tool-call parts, print tool name and args.
                print(f"  Tool Call: {part.tool_name} with args: {getattr(part, 'args', 'N/A')}")
            elif part.part_kind == "tool-return":
                # For tool-return parts, print content if available.
                print(f"  Tool Return: {getattr(part, 'content', 'No content')}")
            else:
                # For system-prompt, user-prompt, text, etc.
                print(f"  {part.part_kind.capitalize()}: {getattr(part, 'content', 'No content')}")
        print("-" * 50)
    print("\n--- End of Steps ---")

if __name__ == "__main__":
    asyncio.run(run_fully_autonomous_agent())