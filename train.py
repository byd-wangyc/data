```python
import os
import csv
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ------------------------------
# 1. 配置参数
# ------------------------------
model_path = "./bge-m3"          # 你的本地 bge-m3 模型路径
train_path = "./data/train.csv"  # 训练数据
output_dir = "./output_model"    # 输出模型保存路径
batch_size = 32
epochs = 3
lr = 2e-5
warmup_ratio = 0.1

# ------------------------------
# 2. 加载模型
# ------------------------------
model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 3. 加载训练数据
# ------------------------------
def load_dataset(csv_path):
    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text1, text2, label = row["text1"], row["text2"], int(row["label"])
            examples.append(InputExample(texts=[text1, text2], label=float(label)))
    return examples

train_examples = load_dataset(train_path)

# ------------------------------
# 4. 构建 DataLoader
# ------------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# ------------------------------
# 5. 定义损失函数（常用对比学习）
# ------------------------------
train_loss = losses.CosineSimilarityLoss(model)

# ------------------------------
# 6. 计算 warmup steps
# ------------------------------
warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

# ------------------------------
# 7. 开始微调
# ------------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': lr},
    output_path=output_dir,
    show_progress_bar=True
)

print(f"✅ 微调完成，模型已保存到: {output_dir}")

# ------------------------------
# 8. 可选：测试编码效果
# ------------------------------
texts = ["发动机盖", "机盖", "刹车片"]
embeddings = model.encode(texts, normalize_embeddings=True)
print("示例 embedding:", embeddings[0][:10])  # 打印前10个维度
```
