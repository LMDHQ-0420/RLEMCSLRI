import os
import json
import torch
import numpy as np
import glob
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    TrainerCallback,
    GPT2Tokenizer
)

class LogicTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id
        self.padding_side = "right"

    def pad(self, features, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors="pt"):
        # features 是一个 List[Dict]，例如 [{'input_ids': [...], 'labels': [...], 'h': ...}, ...]
        
        # 1. 提取各个字段
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # 2. 计算当前 Batch 的最大长度
        curr_max_len = max(len(x) for x in input_ids)
        if pad_to_multiple_of:
            curr_max_len = ((curr_max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        batch_input_ids = []
        batch_labels = []
        
        # 3. 执行 Padding
        for ids, lbls in zip(input_ids, labels):
            pad_len = curr_max_len - len(ids)
            if pad_len > 0:
                batch_input_ids.append(torch.tensor(ids + [self.pad_token_id] * pad_len))
                batch_labels.append(torch.tensor(lbls + [-100] * pad_len))
            else:
                batch_input_ids.append(torch.tensor(ids[:curr_max_len]))
                batch_labels.append(torch.tensor(lbls[:curr_max_len]))

        # 4. 构建输出字典
        batch = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
        }
        
        # 5. 特殊处理：将 'h' 字段从各个样本中提取并堆叠，供 EntropyTrackerTrainer 使用
        if "h" in features[0]:
            batch["h"] = torch.tensor([f["h"] for f in features])
            
        return batch

# ==========================================
# 1. 自定义 Trainer：追踪累计熵并合并评估结果
# ==========================================
class EntropyTrackerTrainer(Trainer):
    def __init__(self, *args, eval_entropy_interval=100000, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_entropy_interval = eval_entropy_interval
        self.cumulative_entropy = 0.0
        self.last_eval_at = 0.0

    def training_step(self, model, inputs, *args, **kwargs):
        # 1. 累加当前 batch 的总熵 (h)
        # 注意：inputs 是一个字典，通过 pop 取出 'h' 以免传给模型导致模型报错
        batch_h = inputs.pop("h", None)
        if batch_h is not None:
            # 累加本 Batch 所有样本的熵值之和
            self.cumulative_entropy += batch_h.sum().item()
        
        # 2. 执行标准训练步 (透传所有接收到的参数)
        loss = super().training_step(model, inputs, *args, **kwargs)
        
        # 3. 检查是否达到 10w 阈值触发评估
        if self.cumulative_entropy - self.last_eval_at >= self.eval_entropy_interval:
            print(f"\n" + "="*50)
            print(f">>> [触发评估] 当前消耗熵总量: {self.cumulative_entropy:.2f} bits")
            self.evaluate()
            self.last_eval_at = self.cumulative_entropy
            print("="*50 + "\n")
            
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 调用原生评估（会遍历字典里的各个 L 桶）
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 计算全量加权平均 Loss
        total_loss = 0.0
        total_samples = 0
        shard_losses = {}
        
        for key, value in metrics.items():
            if "loss" in key and key != f"{metric_key_prefix}_loss":
                shard_name = key.replace(f"{metric_key_prefix}_", "").replace("_loss", "")
                sample_count = metrics.get(f"{metric_key_prefix}_{shard_name}_samples", 0)
                total_loss += value * sample_count
                total_samples += sample_count
                shard_losses[shard_name] = value

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics[f"{metric_key_prefix}_total_avg_loss"] = avg_loss
            metrics[f"{metric_key_prefix}_total_avg_bits"] = avg_loss / np.log(2)
            
            # 漂亮的控制台输出
            print(f"\n" + "="*50)
            print(f"当前累计熵: {self.cumulative_entropy:.2f} bits")
            print(f"全量测试集平均 Loss: {avg_loss:.6f}")
            print(f"全量测试集平均 Bits: {metrics[f'{metric_key_prefix}_total_avg_bits']:.6f}")
            print("-" * 20)
            for shard, val in shard_losses.items():
                print(f"  难度 {shard}: {val:.6f}")
            print("="*50 + "\n")

        return metrics

# ==========================================
# 2. 数据处理
# ==========================================
def preprocess_function(examples):
    inputs = []
    labels = []
    for q, a in zip(examples["q"], examples["a"]):
        combined = q + a
        # 掩码 Q 只有 A 算 Loss
        mask_label = [-100] * len(q) + a 
        inputs.append(combined)
        labels.append(mask_label)
    
    return {
        "input_ids": inputs, 
        "labels": labels, 
        "h": examples["h"] 
    }

# ==========================================
# 3. 运行主函数
# ==========================================
def run_experiment():
    MODEL_SCALE = "30M"
    DATA_ROOT = "data/ECT-Logic"
    
    # 加载配置
    with open("GPT2_Scaling_Logic_Registry.json", "r") as f:
        reg = json.load(f)
    cfg_data = reg["gpt2_scaling_configs"][MODEL_SCALE]
    config = GPT2Config(
        vocab_size=2049, 
        **cfg_data, 
        resid_pdrop=0.0, 
        embd_pdrop=0.0, 
        attn_pdrop=0.0,
        use_cache=True
    )
    model = GPT2LMHeadModel(config)

    # 1. 加载训练集
    train_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "train/hf_shards/shard_*")))
    train_ds = concatenate_datasets([load_from_disk(p) for p in train_paths])
    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=["q", "a"])

    # 2. 加载测试集
    eval_datasets = {}
    for l_val in [2, 5, 10, 15, 20, 25]:
        path = os.path.join(DATA_ROOT, f"test/hf_shards/shard_L{l_val}")
        if os.path.exists(path):
            eval_datasets[f"L{l_val}"] = load_from_disk(path).map(preprocess_function, batched=True, remove_columns=["q", "a"])

    # 3. 训练参数 (修复了 eval_strategy 报错)
    training_args = TrainingArguments(
        output_dir=f"./output_{MODEL_SCALE}",
        num_train_epochs=1,
        per_device_train_batch_size=64,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="no",               # 修复点：evaluation_strategy -> eval_strategy
        save_strategy="no",
        remove_unused_columns=False,      # 必须 False
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        logging_first_step=True
    )

    # 4. 手动构建离线 LogicTokenizer (解决 Network unreachable 问题)
    logic_tokenizer = LogicTokenizer(pad_token_id=0)

    # 5. 实例化自定义 Trainer
    trainer = EntropyTrackerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_datasets,       
        data_collator=logic_tokenizer.pad, # 直接指定 pad 函数作为 collator
        eval_entropy_interval=100000      
    )

    print(f"\n>>> 开始实验：模型规模 {MODEL_SCALE}，每 10w 熵评估一次。")
    trainer.train()
    
    print("\n>>> 训练结束，执行最终完整评估结果：")
    trainer.evaluate()

if __name__ == "__main__":
    run_experiment()