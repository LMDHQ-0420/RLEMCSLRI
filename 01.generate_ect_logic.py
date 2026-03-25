import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool, Manager
import os
import csv
import glob
import networkx as nx
from datasets import Dataset, concatenate_datasets, load_from_disk

# ==========================================
# 1. 核心算法 (逻辑生成与图构建)
# ==========================================

def generate_logic_params_minimal(mu_h, sigma_h, mu_t, sigma_t):
    """采样并微调逻辑参数，确保 d >= 2"""
    target_h = max(2.0, np.random.normal(mu_h, sigma_h))
    target_t = max(target_h * 3.0, np.random.normal(mu_t, sigma_t))
    target_m = int(target_t // 2) 
    
    ratio = target_m / target_h
    best_d_avg = 2.0
    min_diff = float('inf')
    for d_test in np.linspace(2.0, 10.0, 100):
        curr_ratio = d_test / math.log2(d_test)
        if abs(curr_ratio - ratio) < min_diff:
            min_diff = abs(curr_ratio - ratio)
            best_d_avg = d_test
            
    L = max(2, int(round(target_m / best_d_avg)))
    d_sequence = [max(2, target_m // L)] * L
    
    diff_m = target_m - sum(d_sequence)
    for _ in range(abs(diff_m)):
        idx = random.randint(0, L - 1)
        if diff_m > 0: d_sequence[idx] += 1
        elif d_sequence[idx] > 2: d_sequence[idx] -= 1

    for _ in range(200):
        d_sequence = [max(2, d) for d in d_sequence]
        current_h = sum(math.log2(d) for d in d_sequence)
        error = target_h - current_h
        if abs(error) < 0.05: break
        i, j = random.sample(range(L), 2)
        if error > 0:
            if abs(d_sequence[i] - d_sequence[j]) > 1:
                step = 1 if d_sequence[i] < d_sequence[j] else -1
                d_sequence[i] += step; d_sequence[j] -= step
        else:
            if d_sequence[i] > 2: d_sequence[i] -= 1; d_sequence[j] += 1
            elif d_sequence[j] > 2: d_sequence[j] -= 1; d_sequence[i] += 1
                
    return sum(math.log2(d) for d in d_sequence), 2 * sum(d_sequence), L, d_sequence

def calculate_wl_hash(edges):
    """WL 图哈希去重"""
    G = nx.DiGraph(); G.add_edges_from(edges)
    return nx.weisfeiler_lehman_graph_hash(G)

def construct_and_map(h_params, all_token_ids):
    """构建 DAG 逻辑图并映射 Token ID"""
    final_h, final_t, L, d_seq = h_params
    edges = []; path_nodes = list(range(L + 1)) 
    for i in range(L): edges.append((path_nodes[i], path_nodes[i+1]))
    next_node = L + 1
    for i in range(L):
        for _ in range(d_seq[i] - 1):
            edges.append((path_nodes[i], next_node))
            next_node += 1
    
    topo_hash = calculate_wl_hash(edges)
    sampled_ids = random.sample(all_token_ids, next_node)
    random.shuffle(edges)
    
    q_content = []
    for u, v in edges: q_content.extend([sampled_ids[u], sampled_ids[v]])
    q_content.extend([sampled_ids[0], sampled_ids[L]])
    a_content = [sampled_ids[n] for n in path_nodes]
    
    return {"h": round(final_h, 2), "t": final_t, "l": L, "d_seq": str(d_seq), "q": q_content, "a": a_content, "hash": topo_hash}

# ==========================================
# 2. 存储辅助函数
# ==========================================

def worker_task(args):
    mu_h, sigma_h, mu_t, sigma_t, all_token_ids = args
    params = generate_logic_params_minimal(mu_h, sigma_h, mu_t, sigma_t)
    return construct_and_map(params, all_token_ids)

def save_metadata_only(results, base_path):
    csv_dir = os.path.join(base_path, "metadata")
    os.makedirs(csv_dir, exist_ok=True)
    opened_files = {}
    for item in results:
        h_group = int(item['h'])
        if h_group not in opened_files:
            p = os.path.join(csv_dir, f"H{h_group}.csv")
            is_new = not os.path.exists(p)
            f = open(p, "a", newline='')
            writer = csv.writer(f)
            if is_new: writer.writerow(["final_h", "final_t", "L", "d_sequence", "hash"])
            opened_files[h_group] = (f, writer)
        opened_files[h_group][1].writerow([item['h'], item['t'], item['l'], item['d_seq'], item['hash']])
    for f, _ in opened_files.values(): f.close()

# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    BASE_PATH = "data/ECT-Logic"
    TOTAL_SAMPLES = 1000000 
    BATCH_SIZE = 5000       
    SAVE_BLOCK_SIZE = 100000 # 缓冲区大小
    NUM_CORES = 32
    
    os.makedirs(BASE_PATH, exist_ok=True)
    all_token_ids = list(range(50257))
    
    # 彻底禁用 HF 进度条输出
    import datasets
    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bar()
    
    manager = Manager()
    global_hashes = manager.dict()
    
    full_data_buffer = {"q": [], "a": []}
    generated_count = 0
    
    print(f"\n>>> 启动生成系统 [目标: {TOTAL_SAMPLES}]")
    
    with Pool(NUM_CORES) as pool:
        while generated_count < TOTAL_SAMPLES:
            args = (15, 5, 80, 20, all_token_ids)
            batch_raw = pool.map(worker_task, [args] * BATCH_SIZE)
            
            unique_batch = []
            for item in batch_raw:
                if item['hash'] not in global_hashes:
                    global_hashes[item['hash']] = True
                    unique_batch.append(item)
                    full_data_buffer["q"].append(item['q'])
                    full_data_buffer["a"].append(item['a'])
            
            save_metadata_only(unique_batch, BASE_PATH)
            generated_count += len(unique_batch)
            
            efficiency = (len(unique_batch)/BATCH_SIZE)*100
            print(f"进度: {generated_count}/{TOTAL_SAMPLES} | 有效率: {efficiency:.1f}%")

            # --- 每 10 万条存一个文件夹 ---
            if len(full_data_buffer["q"]) >= SAVE_BLOCK_SIZE or generated_count >= TOTAL_SAMPLES:
                if len(full_data_buffer["q"]) > 0:
                    # 分片 ID
                    shard_id = generated_count // SAVE_BLOCK_SIZE
                    shard_path = os.path.join(BASE_PATH, "hf_shards", f"shard_{shard_id}")
                    
                    # 每一个文件夹都是一个标准的数据集分片
                    Dataset.from_dict(full_data_buffer).save_to_disk(shard_path)
                    print(f"--- 成功生成标准数据集分片: {shard_path} ---")
                
                full_data_buffer = {"q": [], "a": []}

    # 删掉后面所有的 concatenate_datasets 逻辑
    print("\n>>> 生成阶段全部完成。")

    # --- 关键修复：从磁盘加载所有分片进行最终合并 ---
    print("\n>>> 正在扫描磁盘分片并执行最终合并...")
    shard_dirs = sorted(glob.glob(os.path.join(BASE_PATH, "hf_shards/shard_*")))
    
    if shard_dirs:
        # 逐个加载已落盘的分片
        loaded_datasets = [load_from_disk(d) for d in shard_dirs]
        final_dataset = concatenate_datasets(loaded_datasets)
        
        final_save_path = os.path.join(BASE_PATH, "hf_dataset_final")
        # 自动分片保存（生成整洁的多个 arrow 文件）
        final_dataset.save_to_disk(final_save_path, max_shard_size="500MB")
        print(f">>> 数据已整合至: {final_save_path}")
    else:
        print(">>> 错误：未发现可合并的分片文件。")

    # ==========================================
    # 4. 绘图部分 (全量分析)
    # ==========================================
    print("\n>>> 正在从 CSV 汇总数据并绘图...")
    csv_files = glob.glob(os.path.join(BASE_PATH, "metadata/H*.csv"))
    if not csv_files:
        print(">>> 未发现元数据 CSV，跳过绘图。")
    else:
        all_h, all_t, all_l = [], [], []
        for f_path in csv_files:
            with open(f_path, 'r') as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    all_h.append(float(row['final_h']))
                    all_t.append(int(row['final_t']))
                    all_l.append(int(row['L']))
        
        plt.figure(figsize=(20, 6))
        # 子图逻辑保持一致
        plt.subplot(1, 3, 1)
        plt.hist(all_h, bins=50, density=True, color='#2ecc71', alpha=0.7, edgecolor='black')
        mu, std = norm.fit(all_h); x = np.linspace(min(all_h), max(all_h), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'r--', lw=2); plt.title(f'H (Entropy)\nmu={mu:.2f}')

        plt.subplot(1, 3, 2)
        plt.hist(all_t, bins=50, density=True, color='#3498db', alpha=0.7, edgecolor='black')
        mu, std = norm.fit(all_t); x = np.linspace(min(all_t), max(all_t), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'r--', lw=2); plt.title(f'T (Tokens)\nmu={mu:.2f}')

        plt.subplot(1, 3, 3)
        plt.hist(all_l, bins=range(min(all_l), max(all_l)+2), color='#f39c12', alpha=0.7, edgecolor='black', align='left')
        plt.title(f'L (Depth)\nRange: {min(all_l)}-{max(all_l)}')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE_PATH, "final_dataset_stats.png"))
        print(f">>> 任务圆满完成。图表已保存。")