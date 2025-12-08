import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

model = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
)


import requests, pathlib

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("./data/Chinook.db")

if not local_path.exists():
    local_path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///data/Chinook.db")

print(f"数据库: {db.dialect}")
print(f"数据表: {db.get_usable_table_names()}")
print(f'艺术家(Artist)数据示例: {db.run("SELECT * FROM Artist LIMIT 5;")}')


from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

system_prompt = """
你是一名专门与SQL数据库交互的智能体。
给定用户的自然语言问题，请先完成以下步骤，再返回答案：
根据输入问题，首先写出语法正确的 {dialect} 查询语句，并只取最多 {top_k} 条记录（除非用户特别指定数量）。
然后执行查询，根据查询结果生成答案。

始终按相关字段排序，返回最具代表性的示例；禁止 SELECT *。

在执行查询之前，必须仔细检查语句的正确性；执行时若报错，重新编写查询并重试。

不要对数据库执行任何 DML 语句（INSERT、UPDATE、DELETE、DROP 等）。

第一步必须先查看数据库中所有表，不可跳过；随后再查询最相关表的表结构。
""".format(dialect=db.dialect, top_k=5)


from langchain.agents import create_agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "sql_db_query": True, # 三种决策 approve, reject, edit
            },
            description_prefix="工具执行待审批" # 设置中断消息提示描述
        )
    ],
    checkpointer=InMemorySaver(),
)


question = "哪个流派的曲目平均时长最长?"
config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    if "__interrupt__" in step:
        print("---有中断消息:---")
        print(step["__interrupt__"])
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])

# 用户输入
user_input = input("请输入指令 (approve/reject/edit): ").strip().lower()
decisions = ["approve", "reject", "edit"]
if user_input in decisions:
    decisions = [{"type": user_input}]
else:
    print("输入指令错误，请输入 approve, reject, edit 中的一个")
    


for step in agent.stream(
    Command(resume={"decisions": decisions}),
    config=config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
