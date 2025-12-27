# -*- coding: utf-8 -*-
# pip install transformers torch faiss-cpu
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import json


class ColBERTReranker:
    """
    基于 transformers 的 ColBERT 重排器
    """
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载 ColBERT 检查点（底层就是 BERT + 线性降维）
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], max_len: int = 128) -> torch.Tensor:
        """
        返回 (batch, seq_len, dim) 的 token 级向量，L2 归一化
        """
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            embs = self.model(**inputs).last_hidden_state   # [B, L, d]
            embs = F.normalize(embs, p=2, dim=-1)           # L2 归一化
        # 去掉 padding token 的向量（可选，保持与官方一致）
        mask = inputs["attention_mask"].unsqueeze(-1)       # [B, L, 1]
        return embs * mask                                  # pad 位置变 0

    def maxsim(self, q_embs: torch.Tensor, d_embs: torch.Tensor) -> float:
        """
        单条查询 vs 单条文档的 MaxSim 得分
        q_embs: [Lq, d]   d_embs: [Ld, d]
        """
        # 1. 点积得到相似度矩阵 [Lq, Ld]
        sim = torch.matmul(q_embs, d_embs.transpose(0, 1))    # [Lq, d] * [d, Ld] = [Lq, Ld]
        # 2. 每个查询 token 取最大
        max_per_qtok, _ = sim.max(dim=1)                      # [Lq]
        # 3. 求和
        return max_per_qtok.sum().item()

    # ===== 离线建索引 =====
    def index_docs(self, docs: List[str], save_path: str, max_len: int = 180):
        """
        把文档预编码成 token 向量并落盘（json + pt）
        """
        os.makedirs(save_path, exist_ok=True)
        for idx, doc in enumerate(docs):
            d_emb = self.encode([doc], max_len=max_len)[0].cpu()  # [Ld, d]
            torch.save(d_emb, os.path.join(save_path, f"{idx}.pt"))
        with open(os.path.join(save_path, "docs.json"), "w", encoding="utf8") as f:
            json.dump(docs, f, ensure_ascii=False)
        print(f"索引完成，共 {len(docs)} 篇文档")

    # ===== 在线重排 =====
    def rerank(self, query: str, index_path: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        返回 [(doc_id, score), ...] 按得分降序
        """
        q_emb = self.encode([query], max_len=32)[0].cpu()           # [Lq, d]
        docs = json.load(open(os.path.join(index_path, "docs.json")))
        scores = []
        for idx in range(len(docs)):
            d_emb = torch.load(os.path.join(index_path, f"{idx}.pt"))
            score = self.maxsim(q_emb, d_emb)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# -------------------- 自测 --------------------
if __name__ == "__main__":
    reranker = ColBERTReranker()

    # 1. 离线建索引（只需一次）
    docs = [
        "ColBERT is a fast and accurate retrieval model.",
        "Late interaction allows token-level matching between query and document.",
        "BERT generates contextualized embeddings for each token."
    ]
    reranker.index_docs(docs, save_path="./colbert_index")

    # 2. 在线重排
    query = "fast retrieval model"
    for idx, score in reranker.rerank(query, "./colbert_index"):
        print(f"doc={idx}  score={score:.3f}")