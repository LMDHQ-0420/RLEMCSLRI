import os
import json
import torch
import numpy as np
import glob
import pandas as pd
from datetime import datetime
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)

# ==========================================
# 1. 进度条增强：在 tqdm 上显示当前参数量规模
# ==========================================
class ScaleInfoCallback(TrainerCallback):
    def __init__(self, scale_name):
        self.scale_name = scale_name

    def on_step_end(self, args, state, control, **kwargs):
        # 实时更新进度条描述，显示当前正在运行的实验规模
        state.trial_name = f"Scale: {self.scale_name}"
        return control

# ==========================================
# 2. 数据处理与 Tokenizer
# ==========================================
class LogicTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id
        self.padding_side = "right"

    def pad(self, features, **kwargs):
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        curr_max_len = max(len(x) for x in input_ids)
        
        batch_input_ids, batch_labels = [], []
        for ids, lbls in zip(input_ids, labels):
            pad_len = curr_max_len - len(ids)
            batch_input_ids.append(torch.tensor(ids + [self.pad_token_id] * pad_len))
            batch_labels.append(torch.tensor(lbls + [-100] * pad_len))

        batch = {"input_ids": torch.stack(batch_input_ids), "labels": torch.stack(batch_labels)}
        if "h" in features[0]:
            batch["h"] = torch.tensor([f["h"] for f in features])
        return batch

def preprocess_function(examples):
    inputs, labels = [], []
    for q, a in zip(examples["q"], examples["a"]):
        inputs.append(q + a)
        labels.append([-100] * len(q) + a)
    return {"input_ids": inputs, "labels": labels, "h": examples["h"]}

# ==========================================
# 3. 自定义 Trainer：增加 QA 和 Token 计数
# ==========================================
class SilentEntropyTrainer(Trainer):
    def __init__(self, *args, eval_entropy_interval=100000, csv_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_entropy_interval = eval_entropy_interval
        self.csv_dir = csv_dir
        
        # 核心计数器
        self.cumulative_entropy = 0.0
        self.cumulative_qa = 0          # 已训练的 QA 总数
        self.cumulative_tokens = 0      # 已训练的有效 Token 总数 (不含 padding)
        
        self.last_eval_at = 0.0

    def training_step(self, model, inputs, *args, **kwargs):
        # 1. 统计逻辑熵
        batch_h = inputs.pop("h", None)
        if batch_h is not None:
            self.cumulative_entropy += batch_h.sum().item()
        
        # 2. 统计 QA 数量 (Batch Size)
        self.cumulative_qa += inputs["input_ids"].size(0)
        
        # 3. 统计有效 Token 数量
        # labels 中 -100 代表 padding 或不需要计算 loss 的部分
        # inputs["input_ids"] 中非 0 (pad_token_id) 的部分为有效 token
        # 这里建议统计 input_ids 中非 pad 的数量，即实际喂入模型的 token 数
        valid_tokens = (inputs["input_ids"] != 0).sum().item()
        self.cumulative_tokens += valid_tokens
        
        loss = super().training_step(model, inputs, *args, **kwargs)
        
        # 检查是否达到评估间隔
        if self.cumulative_entropy - self.last_eval_at >= self.eval_entropy_interval:
            self.evaluate()
            self.last_eval_at = self.cumulative_entropy
        return loss

    def log(self, logs):
        pass

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        for key, value in metrics.items():
            if "loss" in key and key != f"{metric_key_prefix}_loss":
                shard_name = key.replace(f"{metric_key_prefix}_", "").replace("_loss", "")
                csv_path = os.path.join(self.csv_dir, f"{shard_name}.csv")
                
                # 在 DataFrame 中增加两列
                new_data = pd.DataFrame([{
                    "cumulative_entropy": self.cumulative_entropy,
                    "cumulative_qa": self.cumulative_qa,         # 新增列
                    "cumulative_tokens": self.cumulative_tokens, # 新增列
                    "eval_loss": value,
                    "eval_bits": value / np.log(2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])
                new_data.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        return metrics

# ==========================================
# 4. 运行主函数
# ==========================================
def run_experiment():
    DATA_ROOT = "data/ECT-Logic"
    REGISTRY_FILE = "GPT2_Scaling_Logic_Registry.json"
    EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(REGISTRY_FILE, "r") as f:
        reg = json.load(f)
    
    configs = reg["gpt2_scaling_configs"]
    global_settings = reg["global_settings"]

    # 1. 加载训练集
    train_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "train/hf_shards/shard_*")))
    train_ds = concatenate_datasets([load_from_disk(p) for p in train_paths]).map(
        preprocess_function, batched=True, remove_columns=["q", "a"]
    )

    # 2. 遍历加载所有连续的 L 难度 (2-25)
    eval_datasets = {}
    print(">>> 正在初始化测试集 (L2 - L25)...")
    for l_val in range(2, 26): # 包含 2 到 25
        path = os.path.join(DATA_ROOT, f"test/hf_shards/shard_L{l_val}")
        if os.path.exists(path):
            eval_datasets[f"L{l_val}"] = load_from_disk(path).map(
                preprocess_function, batched=True, remove_columns=["q", "a"]
            )

    logic_tokenizer = LogicTokenizer(pad_token_id=0)

    # 3. 循环实验
    for scale_name, cfg_data in configs.items():
        exp_dir = f"output/{EXPERIMENT_TIME}/{scale_name}"
        os.makedirs(exp_dir, exist_ok=True)

        config = GPT2Config(vocab_size=global_settings["vocab_size"], **cfg_data, use_cache=False)
        model = GPT2LMHeadModel(config)

        training_args = TrainingArguments(
            output_dir=exp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=64,
            learning_rate=5e-4,
            lr_scheduler_type="cosine",
            eval_strategy="no",
            save_strategy="no",
            logging_steps=9999999,      # 彻底屏蔽标准日志
            report_to="none",
            disable_tqdm=False,        # 仅保留 TQDM
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        )

        trainer = SilentEntropyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_datasets,
            data_collator=logic_tokenizer.pad,
            eval_entropy_interval=100000,
            csv_dir=exp_dir,
            callbacks=[ScaleInfoCallback(scale_name)] # 添加参数量显示回调
        )

        trainer.train()
        trainer.evaluate() # 最终评估
        
        # 释放显存
        del model, trainer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_experiment()