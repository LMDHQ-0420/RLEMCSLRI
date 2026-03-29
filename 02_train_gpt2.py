import os
import json
import torch
import numpy as np
import glob
import pandas as pd
import gc
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

# ==========================================
# 1. 数据对齐函数 (保持不变)
# ==========================================
def logic_data_collator(batch, pad_token_id=0):
    input_ids, labels, h_values, a_lens = [], [], [], []
    for item in batch:
        q, a = item["question"], item["answer"]
        input_ids.append(torch.tensor(q + a))
        labels.append(torch.tensor([-100] * len(q) + a))
        h_values.append(item.get("h", 0.0))
        a_lens.append(len(a))
    max_len = max(len(x) for x in input_ids)
    padded_ids = torch.stack([torch.cat([ids, torch.full((max_len - len(ids),), pad_token_id)]) for ids in input_ids])
    padded_labels = torch.stack([torch.cat([lbls, torch.full((max_len - len(lbls),), -100)]) for lbls in labels])
    return {
        "input_ids": padded_ids, 
        "labels": padded_labels, 
        "h": torch.tensor(h_values, dtype=torch.float32),
        "a_lens": torch.tensor(a_lens, dtype=torch.int32)
    }

# ==========================================
# 2. 核心实验执行函数 (非累积式，每次 N 重新训练)
# ==========================================
def run_convergence_experiment(
    config_data, # 传入配置以便重新初始化模型
    global_settings,
    dataloader,
    device,
    search_log_path,
    model_save_path,
    target_acc=1.0,
    max_inner_epochs=10000,
    loss_plateau_min_delta=1e-6,
    loss_plateau_patience=100, # 独立训练难度大，建议增加耐心值
):
    total_qa_count = 0      
    saturation_stats = None
    first_saturation_found = False

    if os.path.exists(search_log_path): os.remove(search_log_path)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 这里的 current_pool 会不断变大，但我们每次都重开模型训练整个 pool
    current_pool = []

    for batch in tqdm(dataloader, desc="  Increasing N", leave=False):
        current_pool.append(batch)
        total_qa_count += batch["input_ids"].size(0)
        
        # --- 核心修改：针对当前的 total_qa_count，重新初始化模型 ---
        model_config = GPT2Config(vocab_size=global_settings["vocab_size"], **config_data)
        model = GPT2LMHeadModel(model_config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        
        model.train()
        converged = False
        best_loss = float("inf")
        no_improve_count = 0
        last_metrics = {}

        # 内部训练循环
        for inner_epoch in range(max_inner_epochs):
            epoch_correct = 0
            epoch_total = 0
            epoch_loss_sum = 0.0
            epoch_loss_steps = 0
            
            # 训练当前 Pool 里的所有数据
            for train_step in current_pool:
                input_ids = train_step["input_ids"].to(device)
                labels = train_step["labels"].to(device)
                
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss_sum += loss.item()
                epoch_loss_steps += 1

                with torch.no_grad():
                    shift_logits = outputs.logits[:, :-1, :]
                    shift_labels = labels[:, 1:]
                    answer_mask = (shift_labels != -100)
                    preds = torch.argmax(shift_logits, dim=-1)
                    token_correct = (preds == shift_labels) | (~answer_mask)
                    seq_correct = token_correct.all(dim=1).sum().item()
                    epoch_correct += seq_correct
                    epoch_total += input_ids.size(0)

            current_loss = epoch_loss_sum / epoch_loss_steps
            current_acc = epoch_correct / epoch_total

            last_metrics = {
                "n": total_qa_count,
                "inner_epoch": inner_epoch,
                "loss": current_loss,
                "acc": current_acc,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

            if (best_loss - current_loss) > loss_plateau_min_delta:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if current_acc >= target_acc:
                converged = True
                break
            
            # 如果在当前 N 下无法收敛，提前终止此轮 N 的训练
            if no_improve_count >= loss_plateau_patience:
                tqdm.write(f"    [FAIL] N={total_qa_count} could not converge. Moving to next N.")
                break
        
        # 记录该 N 阶段的最终结果
        if last_metrics:
            pd.DataFrame([last_metrics]).to_csv(search_log_path, mode='a', index=False, header=not os.path.exists(search_log_path))
        
        # 只有在收敛的情况下，才考虑保存第一个饱和点模型
        if converged and not first_saturation_found:
            first_saturation_found = True
            model.save_pretrained(model_save_path)
            tqdm.write(f"    [SATURATION] N={total_qa_count} reached 100% Acc.")
            
            temp_tokens = sum(b["a_lens"].sum().item() for b in current_pool)
            temp_h = sum(b["h"].sum().item() for b in current_pool)
            saturation_stats = {
                "n": total_qa_count,
                "tokens": temp_tokens,
                "nll_bits": (current_loss * temp_tokens) / np.log(2),
                "h_bits": temp_h
            }
        
        # 必须显式清理内存，因为每一轮 N 都开了新模型
        del model; del optimizer
        torch.cuda.empty_cache()
        gc.collect()

    return saturation_stats if saturation_stats else {"n": total_qa_count, "tokens": 0, "nll_bits": 0, "h_bits": 0}

# ==========================================
# 3. 主程序逻辑 (适配非累积函数调用)
# ==========================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG_PATH = "gpt2_reg.json"
    DATA_ROOT = "data"
    OUTPUT_BASE = "outputs"
    
    with open(CONFIG_PATH, "r") as f:
        reg = json.load(f)
    
    configs = reg["gpt2_scaling_configs"]
    global_settings = reg["global_settings"]
    
    for scale_name, cfg_data in configs.items():
        save_dir = os.path.join(OUTPUT_BASE, f"GPT2_{scale_name}")
        search_dir = os.path.join(save_dir, "search_saturation")
        model_base_dir = os.path.join(save_dir, "models")
        os.makedirs(search_dir, exist_ok=True)
        
        q_len_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "ECT-Logic", "Q_LEN_*")), 
                            key=lambda x: int(os.path.basename(x).split("_")[2]))

        for q_dir in q_len_dirs:
            q_len = int(os.path.basename(q_dir).split("_")[2])
            
            # --- Random 实验 ---
            rand_train_files = glob.glob(os.path.join(DATA_ROOT, "ECT-Random", f"Q_LEN_{q_len}", "train/*.parquet"))
            if rand_train_files:
                loader = DataLoader(load_dataset("parquet", data_files=rand_train_files, split="train"), 
                                   batch_size=32, shuffle=True, collate_fn=logic_data_collator)
                log_p = os.path.join(search_dir, f"random_qlen_{q_len}_process.csv")
                model_p = os.path.join(model_base_dir, f"random_qlen_{q_len}")
                
                # 传入配置参数，内部会重新初始化模型
                res = run_convergence_experiment(cfg_data, global_settings, loader, DEVICE, log_p, model_p)
                
                pd.DataFrame([{"q_len": q_len, "saturation_n": res["n"], "nll_bits": res["nll_bits"]}]).to_csv(
                    os.path.join(save_dir, "random_results.csv"), mode='a', index=False, header=not os.path.exists(os.path.join(save_dir, "random_results.csv")))

            # --- Logic 实验 ---
            logic_train_files = glob.glob(os.path.join(q_dir, "train/*.parquet"))
            if logic_train_files:
                loader = DataLoader(load_dataset("parquet", data_files=logic_train_files, split="train"), 
                                   batch_size=32, shuffle=True, collate_fn=logic_data_collator)
                log_p = os.path.join(search_dir, f"logic_qlen_{q_len}_process.csv")
                model_p = os.path.join(model_base_dir, f"logic_qlen_{q_len}")
                
                res = run_convergence_experiment(cfg_data, global_settings, loader, DEVICE, log_p, model_p)
                
                pd.DataFrame([{"q_len": q_len, "saturation_n": res["n"], "nll_bits": res["nll_bits"], "h_bits": res["h_bits"]}]).to_csv(
                    os.path.join(save_dir, "logic_results.csv"), mode='a', index=False, header=not os.path.exists(os.path.join(save_dir, "logic_results.csv")))

if __name__ == "__main__":
    main()