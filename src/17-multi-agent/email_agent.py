import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool


load_dotenv()

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
    "请用中文回复"
)

@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    print("SubAgent email_agent's tool: send_email is called")
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


email_agent = create_agent(
    model=ChatOpenAI(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
    ),
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)

def test_email_agent():
    query = "Send a email to the design team ['design-team@example.com'] about reviewing the design document"
    # query = "hello"
    for step in email_agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        # stream_mode="updates", # 默认值是updates, 可以省略
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    test_email_agent()
