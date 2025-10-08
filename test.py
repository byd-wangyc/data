```python
import os
import csv
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr

# ------------------------------
# 1. 配置参数
# ------------------------------
model_path = "./bge-m3"          # 预训练模型（本地路径）
train_path = "./data/train.csv"  # 训练集
test_path = "./data/test.csv"    # 测试集
output_dir = "./output_model"    # 输出微调后模型路径
batch_size = 32
epochs = 3
lr = 2e-5
warmup_ratio = 0.1

# ------------------------------
# 2. 加载模型
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_path, device=device)

# ------------------------------
# 3. 数据加载函数
# ------------------------------
def load_dataset(csv_path):
    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text1, text2, label = row["text1"], row["text2"], float(row["label"])
            examples.append(InputExample(texts=[text1, text2], label=label))
    return examples

train_examples = load_dataset(train_path)
test_examples = load_dataset(test_path)

# ------------------------------
# 4. 构建 DataLoader
# ------------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model)

# ------------------------------
# 5. 定义 warmup
# ------------------------------
warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

# ------------------------------
# 6. 定义评估函数
# ------------------------------
def evaluate(model, examples):
    """计算平均余弦相似度与真实标签的相关性"""
    texts1 = [ex.texts[0] for ex in examples]
    texts2 = [ex.texts[1] for ex in examples]
    labels = np.array([ex.label for ex in examples])

    emb1 = model.encode(texts1, normalize_embeddings=True, batch_size=64)
    emb2 = model.encode(texts2, normalize_embeddings=True, batch_size=64)
    sims = np.sum(emb1 * emb2, axis=1)  # 余弦相似度

    corr, _ = spearmanr(labels, sims)  # 计算斯皮尔曼相关系数
    auc = roc_auc_score(labels, sims) if len(set(labels)) > 1 else None

    # 用 0.5 阈值粗略计算准确率
    acc = accuracy_score(labels, (sims > 0.5).astype(int))
    return {"spearman": corr, "auc": auc, "accuracy": acc}

# ------------------------------
# 7. 微调前的评估
# ------------------------------
print("🔍 微调前模型评估中...")
before_metrics = evaluate(model, test_examples)
print(f"微调前 - Spearman: {before_metrics['spearman']:.4f}, "
      f"AUC: {before_metrics['auc']:.4f}, ACC: {before_metrics['accuracy']:.4f}")

# ------------------------------
# 8. 开始微调
# ------------------------------
print("🚀 开始微调...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': lr},
    output_path=output_dir,
    show_progress_bar=True
)

# ------------------------------
# 9. 加载微调后模型再评估
# ------------------------------
model_finetuned = SentenceTransformer(output_dir, device=device)
print("🔍 微调后模型评估中...")
after_metrics = evaluate(model_finetuned, test_examples)

print(f"✅ 微调后 - Spearman: {after_metrics['spearman']:.4f}, "
      f"AUC: {after_metrics['auc']:.4f}, ACC: {after_metrics['accuracy']:.4f}")

# ------------------------------
# 10. 对比结果
# ------------------------------
print("\n📊 微调效果对比：")
print(f"Spearman 相关提升: {after_metrics['spearman'] - before_metrics['spearman']:.4f}")
print(f"AUC 提升: {after_metrics['auc'] - before_metrics['auc']:.4f}")
print(f"准确率提升: {after_metrics['accuracy'] - before_metrics['accuracy']:.4f}")
```
