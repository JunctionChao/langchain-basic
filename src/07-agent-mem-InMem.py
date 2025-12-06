import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()


model = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver() # 内存存储历史消息
)

# config 中 thread_id用来区分不同会话
config = {"configurable": {"thread_id": "1"}}

# 第一轮对话
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首五言绝句诗，不要多余解释"}]},
    config=config
)
# 第二轮对话
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "将上面的诗进行解析"}]},
    config=config
)
for message in resp["messages"]: # 同一会话(thread_id)的所有历史消息
    message.pretty_print()

