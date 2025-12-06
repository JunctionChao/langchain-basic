import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver


load_dotenv()

model = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
)

DB_URI = os.getenv("DB_URI")
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,  
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
        {"messages": [{"role": "user", "content": "用一句话将上面的诗进行分析总结"}]},
        config=config
    )
    for message in resp["messages"]: # 同一会话(thread_id)的所有历史消息
        message.pretty_print()

