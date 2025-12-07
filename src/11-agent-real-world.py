import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy


load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """
你是一个诙谐的天气预测助手，你会用幽默的方式回答用户的问题。
你有两个可以使用的工具:
- get_weather_for_location: 通过城市名称获取天气信息
- get_user_location: 获取用户当前位置
"""

# 工具定义
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 上下文对象，提供一些静态信息，用户 ID、数据库连接或其他代理调用的依赖关系
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "上海" if user_id == "1" else "北京"


# 定义结构化输出格式
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# 记忆存储/检查点
checkpointer = InMemorySaver()

agent = create_agent(
    model=ChatOpenAI(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3, timeout=30, max_tokens=96e3, max_retries=5
    ),
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context, # 提供一个上下文schema
    response_format=ToolStrategy(ResponseFormat),
)



config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "现在室外天气如何?"}]},
    config=config,
    context=Context(user_id="1") # 上下文数据传入
)
print(response['structured_response'])
# ResponseFormat(punny_response="It's always sunny in 上海! 上海今天的天气好到连太阳都在加班，云朵都怕晒黑躲起来了！", weather_conditions='sunny')

response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢!"}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])
# ResponseFormat(punny_response='不客气！希望您在上海的晴天里心情也像天气一样明媚灿烂！记得涂防晒霜哦，毕竟太阳这么努力工作！☀️', weather_conditions='sunny')
