from langchain_text_splitters import SpacyTextSplitter  # pip install spacy
from langchain_community.document_loaders import PyPDFLoader
import re


def normalize_text_for_chunking(text: str) -> str:
    """
    预处理文本，消除空行和不合理换行，适合后续语义分块。
    规则：
      1. 移除开头/结尾空白；
      2. 将连续空白（包括换行、制表符、空格）统一替换为单个空格；
      3. 特别处理中文标点后不应换行的情况（可选增强）。
    """
    # 步骤1：标准化所有空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 步骤2：修复常见中文断行问题（可选，增强语义连贯性）
    # 例如：避免“为\n138” → “为 138”，但实际已由 \s+ 处理
    # 如需更精细控制，可添加：
    # text = re.sub(r'([^。！？\n])\n([^。！？\n])', r'\1 \2', text)  # 但 \s+ 已足够
    return text.strip()


# 按页加载pdf文件    List[Document]
file_path = "../../../src/pdfs/健康档案.pdf"
loader = PyPDFLoader(file_path, mode='page') # mode in ['page', 'single']
documents = loader.load()

# 对文档内容进行预处理
for doc in documents:
    doc.page_content = normalize_text_for_chunking(doc.page_content)

# 使用 SpacyTextSplitter 切分中文
splitter = SpacyTextSplitter(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
    pipeline="zh_core_web_sm" # 中文分词
)

chunks = splitter.split_documents(documents)

print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
# print(chunks[0].page_content)
for i, chunk in enumerate(chunks[:5]):
    print("-" * 80)
    # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
    print(f'Chunk {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')



# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,    # 每个文本块的最大字符数
#     chunk_overlap=200,  # 块之间的重叠字符数
#     add_start_index=True,  # 添加起始索引
# )

# splits = text_splitter.split_documents(documents)
# print(len(splits), type(splits[0]))  # 10  <class 'langchain_core.documents.base.Document'>
# print(splits[0].page_content)


# 1.文本分块主要基于两大考量：模型的上下文限制和检索生成的性能需求。
#     文本块的大小必须小于等于嵌入模型的上下文窗口。
# 2.嵌入过程：分词 -> 向量化 -> 池化。文本块越长，包含的语义点越多，单一向量所承载的信息就越稀释，导致其表示变得笼统，关键细节被模糊化，从而降低了检索的精度。
# 3. LLM处理非常长的、充满大量信息的上下文时，它倾向于更好地记住开头和结尾的信息，而忽略中间部分的内容
# 4. 一个好的文本块应该聚焦于一个明确、单一的主题。
# 5. 常用分块策略
#     5.1 固定大小分块：将文本均匀切分成固定大小的块，简单直接。
#     5.2 递归分块：基于递归算法，根据句子、段落或段落标题等逻辑单位进行切分，保持逻辑连贯性。
#     5.3 语义分块：利用嵌入模型的语义理解能力，将文本切分成基于语义相似度的块，确保每个块聚焦于一个主题。
#     5.4 文档结构分块：根据文档的逻辑结构（如章节、段落、标题等）进行切分，保持结构完整性。

# 使用PyPDFLoader加载pdf文件，对文档内容进行预处理（消除空行和不合理换行），然后使用
# SpacyTextSplitter(
#     separator="\n",
#     chunk_size=600,
#     chunk_overlap=100,
#     pipeline="zh_core_web_sm" # 中文分词
# )
# 对中文文档进行分块，保持句子的完整性，避免切分在中间断开。