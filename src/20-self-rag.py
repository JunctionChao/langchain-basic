# Self-RAG: rag中引入迭代推理和自我评估，以提高回答的准确性和相关性。
# 关键步骤：
#   1.根据问题判断是否需要检索（避免不必要的检索，动态适应）
#   2.检索到的文件是否相关（对每份检索的文档评估相关性，如果无用则丢弃）
#   3.答复是否解决用户问题（即使答案事实正确，也可能无法完全回答用户的问题。给定答案和用户问题后，系统会预测最终答案是否有用，并决定是否再生或停止）
# 参考：https://www.datacamp.com/tutorial/self-rag


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
import os
from typing import List
from typing_extensions import TypedDict


RETRIEVAL_MAX_COUNT = 2 # 最大检索次数
REGENERATION_MAX_COUNT = 2 # 最大生成次数


load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.3, timeout=30, max_tokens=96e3, max_retries=5
)
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
    model="bge-large:latest",
    check_embedding_ctx_length=False
)
vector_store = Chroma(
    collection_name="example_archive_with_bge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vector_store.as_retriever() # 文档检索器


# 创建一个结构化的 LLM，用于根据预定义的模式生成结构化输出
def create_structured_llm(schema):
    # llm = ChatOpenAI(model=model, temperature=0)
    # return llm.with_structured_output(schema)

    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    # 尝试使用PydanticOutputParser，如果失败则使用更灵活的方式
    def parse_output(output):
        # 如果输出已经是Pydantic对象，直接返回
        if isinstance(output, schema):
            return output
        
        # 如果输出是字符串，尝试解析
        if isinstance(output, str):
            output = output.strip()
            # 如果是原始的"yes"或"no"，直接创建对象
            if output.lower() in ["yes", "no"]:
                return schema(binary_score=output)
            # 否则尝试JSON解析
            try:
                import json
                parsed = json.loads(output)
                return schema(**parsed)
            except json.JSONDecodeError:
                # 如果JSON解析失败，回退到"yes"或"no"判断
                if "yes" in output.lower():
                    return schema(binary_score="yes")
                elif "no" in output.lower():
                    return schema(binary_score="no")
                else:
                    # 无法解析，默认返回"no"
                    return schema(binary_score="no")
        
        # 其他情况，尝试直接转换
        try:
            return schema(**output)
        except:
            return schema(binary_score="no")
    
    return llm | parse_output

# 简单的二值评分模型，用于评估回答是否符合问题的要求
class BinaryScoreModel(BaseModel):
    binary_score: str = Field(..., description="Binary score: 'yes' or 'no'")
    
    # @model_validator(mode='before')
    # @classmethod
    # def handle_raw_string(cls, v):
    #     # 处理原始字符串输入
    #     if isinstance(v, str):
    #         return {'binary_score': v.strip()}
    #     return v

# 提示词模板
def create_grading_prompt(system_message, human_template):
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template),
    ])

# 1.检索评估器 判断检索到的文档是否与用户问题相关
retrieval_evaluator_llm = create_structured_llm(BinaryScoreModel)
retrieval_evaluator_prompt = create_grading_prompt(
    "你是文档检索评估员，负责检查检索到的文档与用户问题的相关性。若文档包含与问题相关的关键词或语义含义，则评定其为相关文档。输出评分“yes”或“no”。",
    "检索到的文档: {document} \n\n 用户问题: {question}"
)
retrieval_grader = retrieval_evaluator_prompt | retrieval_evaluator_llm

# 2.幻觉评估器 判断答复是否基于检索事实
hallucination_grader = create_grading_prompt(
    "你是一位评分员，负责评估大模型生成内容是否基于检索的事实集。请给出评分'yes'或'no'。'yes'表示该答案基于事实集支撑",
    "检索事实集: {documents} \n\n 大模型生成内容: {generation}"
) | create_structured_llm(BinaryScoreModel)

# 3.答案评估器 判断答复是否解决用户问题
answer_grader = create_grading_prompt(
    "你是一位专业评分员，负责评估答案是否解决用户问题。请给出评分'yes'或'no'。'yes'表示该答案解决了问题",
    "用户问题: {question} \n\n 答案: {generation}"
) | create_structured_llm(BinaryScoreModel)

question_rewriter = create_grading_prompt(
    "你是一个专业的问题重写器，负责将用户问题转换为更优化的版本，以提高向量存储检索的效果。",
    "最初问题: {question} \n 构造一个更优化的问题，提高检索效果。"
) | llm | StrOutputParser()


# from langsmith import Client
# client = Client()
# rag_prompt = client.pull_prompt("rlm/rag-prompt") # input_variables=['context', 'question']
# def format_docs(docs):
#     return "\\n\\n".join(doc.page_content for doc in docs)
rag_prompt = create_grading_prompt(
    "你是一个问答助手，请利用以下检索到的上下文内容回答问题。若不知答案，请直接表明未知。答案最多使用三句话，保持简洁",
    "上下文: {context} \n\n 用户问题: {question}"
)
rag_chain = rag_prompt | llm | StrOutputParser()


# 定义状态图
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        retrieval_count: retrieve num maximum
    """

    question: str
    generation: str
    documents: List[str]
    retrieval_count: int # 检索次数
    regen_count: int # 生成次数


# 检索节点
def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    retrieval_count = state.get("retrieval_count", 0) + 1
    print(f"---RETRIEVE COUNT {retrieval_count}---")
    return {"documents": documents, "question": question, "retrieval_count": retrieval_count}

# 答案生成节点
def generate(state: GraphState):
    regen_count = state.get("regen_count", 0) + 1
    print(f"---GENERATE COUNT {regen_count}---")
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": rag_chain.invoke({"context": state["documents"], "question": state["question"]}),
        # "retrieval_count": state["retrieval_count"],
        "regen_count": regen_count
    }

# 文档评估节点
def grade_documents(state: GraphState):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    if state["retrieval_count"] >= RETRIEVAL_MAX_COUNT:
        return {"documents": documents, "question": question, "retrieval_count": state["retrieval_count"]}

    filtered_docs = []
    for d in documents:
        grade = retrieval_grader.invoke({"question": question, "document": d.page_content}).binary_score
        # res = retrieval_grader.invoke({"question": question, "document": d.page_content})
        # grade = res.binary_score
        print(f"CURRENT DOCUMENT: {d.page_content}")
        print(f"---GRADE: DOCUMENT {'RELEVANT' if grade == 'yes' else 'NOT RELEVANT'}---")
        if grade == "yes":
            filtered_docs.append(d)

    return {"documents": filtered_docs, "question": question, "retrieval_count": state["retrieval_count"]}

# 问题重写节点
def transform_query(state: GraphState):
    print("---TRANSFORM QUERY---")
    question_rewrite = question_rewriter.invoke({"question": state["question"]})
    print(f"---QUESTION REWRITE: {question_rewrite}---")
    return {"documents": state["documents"], "question": question_rewrite, "retrieval_count": state["retrieval_count"]}
    
# edge function 
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS 文档评估中---")
    if not state["documents"]:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY  检索文档不相关，问题重写---")
        return "transform_query"
    print("---DECISION: GENERATE  检索文档相关，答案生成---")
    return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS 验证是否出现幻觉---")
    hallucination_score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]}).binary_score
    if hallucination_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS  幻觉检测通过---")
        answer_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]}).binary_score
        if answer_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION  回答解决了问题---")
            return "useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION  回答未解决问题---")
        if state["retrieval_count"] >= RETRIEVAL_MAX_COUNT:
            return "useful"
        return "not useful"
    if state["regen_count"] >= REGENERATION_MAX_COUNT:
        return "useful"
    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY  幻觉检测未通过，重新生成---")
    return "not supported"


from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)
# add node
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
# add edge
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate, # decide_to_generate函数返回值为节点名表明路径，或者用返回值做映射到节点名
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "transform_query",
        "not supported": "generate",
    }
)

app_self_rag = workflow.compile()

def save_graph_visualization(graph, filename: str = "graph.png") -> None:  
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
    except IOError as e:
        print(f"Failed to save graph visualization: {e}")


if __name__ == "__main__":
    # save_graph_visualization(app_self_rag, "imgs/app_self_rag.png")
    res = app_self_rag.invoke({"question": "患者张三九的基本信息？"})
    # print(res)
    print(res["generation"])


# self rag的局限性
# 计算速度和成本。反复的检索和精炼周期需要大量处理能力，尤其在实时应用中会拖慢速度
# 依赖固定知识库而非网络搜索，当所需信息不可得时，可能会遇到困难
# 幻觉检测。尽管有幻觉检测机制，但在复杂场景下可能会出现错误判断
