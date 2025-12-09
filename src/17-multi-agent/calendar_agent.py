import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool


load_dotenv()

CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
    "请用中文回复"
)

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    print("SubAgent calendar_agent's tool: create_calendar_event is called")
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    print("SubAgent calendar_agent's tool: get_available_time_slots is called")
    return ["09:00", "14:00", "16:00"]

calendar_agent = create_agent(
    model=ChatOpenAI(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
    ),
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)


def test_calendar_agent():
    query = "Schedule a named Design Team meeting ['zhangsan@example.com', 'lisi@example.com'] on 2025-12-10 at 2pm for 1 hour"
    # query = "hello"
    for step in calendar_agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        # stream_mode="updates", # 默认值是updates, 可以省略
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    test_calendar_agent()