import os
# from langchain_deepseek import ChatDeepSeek
from langchain_qwq import ChatQwen
from langchain_community.document_loaders import BiliBiliLoader
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0), # 分钟
        }
        
        doc.metadata = metadata
        # print(doc)
        bili.append(doc)
        
except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)


# 3. 配置元数据字段信息
metadata_field_info = [
    {"name": "title", "description": "视频标题", "type": "string"},
    {"name": "author", "description": "视频作者", "type": "string"},
    {"name": "view_count", "description": "视频观看次数", "type": "integer"},
    {"name": "length", "description": "视频时间长度", "type": "integer"},
]

# 4. 创建自查询检索器
# llm = ChatDeepSeek(
#     model="deepseek-chat", 
#     temperature=0, 
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )
llm = ChatQwen(
    model="qwen-plus", 
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    results = retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的视频")
