import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

load_dotenv()

# 初始化 LLM
llm = ChatQwen(model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"))


# 1. 定义数据结构
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

print(f"PersonInfo schema: {PersonInfo.model_json_schema()}")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建提示模板
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}, # 构建格式指令
)

# # 打印格式指令
print("\n--- Format Instructions ---")
print(parser.get_format_instructions())
print("--------------------------\n")

# 4. 创建处理链
chain = prompt | llm | parser

# 5. 定义输入文本并执行调用链
text = "张三今年30岁，他擅长Python和Go语言。"
result = chain.invoke({"text": text})

# 6. 打印结果
print("\n--- 解析结果 ---")
print(f"结果类型: {type(result)}")
print(result)
print("--------------------\n")

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"技能: {result.skills}")
