import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.vocab_size, type(tokenizer).__name__)
model = AutoModel.from_pretrained(model_name)


def encode(texts: list[str], max_len: int = 128) -> torch.Tensor:
    """
    返回 (batch, seq_len, dim) 的 token 级向量，L2 归一化
    """
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,     # 根据具体情况设置一条序列允许有多少个 token， 让 95 % 的样本刚好不截断
        return_tensors="pt",
    ) # text -> token ids
    print(f"token ids shape: {inputs.input_ids.shape}")             # [B, L]
    print(f"token ids:\n {inputs.input_ids}")
    # print(f"attention mask:\n {inputs.attention_mask}")
    print(f"attention mask shape: {inputs.attention_mask.shape}")

    model.eval()
    with torch.no_grad():
        embs = model(**inputs).last_hidden_state  # [B, L, d]
        embs = F.normalize(embs, p=2, dim=-1)  # 在最后一个维度上 L2 归一化  [B, L, d]
        print(f"embs[B, L, d] 最后一个维度L2归一化后的  shape: {embs.shape}")
        print(f"mask之前:\n {embs[0, 5, :][:10]}")
    # 去掉 padding token 的向量（可选，保持与官方一致）
    mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]    unsqueeze(-1)扩展一个维度，从 [B, L] 变成 [B, L, 1]
    print(f"mask shape: {mask.shape}")
    return embs * mask  # pad 位置变 0


if __name__ == "__main__":
    # texts = ["hello world", "hello langchain"]
    # embs = encode(texts)
    # print(embs.shape) # [2, 128, 768]   [B, L, d]
    # print(f"mask之后:\n {embs[0, 5, :][:10]}")

    print("--" * 20)
    sim = torch.tensor([[3, 1, 4],
                    [2, 5, 0]])   # shape [2, 3]
    max_val, max_idx = sim.max(dim=1) # 在第 1(0开始) 个维度上取每个位置的最大值 (行方向)
    print(max_val)   # tensor([4, 5])
    print(max_idx)   # tensor([2, 1])