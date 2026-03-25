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
from datasets import Dataset

# ==========================================
# 1. 核心算法 (逻辑生成与图构建)
# ==========================================

def generate_logic_params_fixed_l(target_l, mu_h, sigma_h, mu_t, sigma_t):
    """
    专门为指定 L 生成参数。
    通过随机调整节点的出度序列，在固定 L 的前提下，使 H 和 T 尽量符合目标分布。
    """
    target_h = max(2.0, np.random.normal(mu_h, sigma_h))
    # 确保 target_t 足以支撑逻辑深度
    target_t = max(target_l * 4.0, np.random.normal(mu_t, sigma_t))
    target_m = int(target_t // 2)
    
    L = target_l
    d_avg = max(2, target_m // L)
    d_sequence = [d_avg] * L
    
    # 迭代微调出度序列以逼近目标熵 H
    for _ in range(300):
        d_sequence = [max(2, d) for d in d_sequence]
        current_h = sum(math.log2(d) for d in d_sequence)
        error = target_h - current_h
        if abs(error) < 0.05: break
        
        i = random.randint(0, L - 1)
        if error > 0: d_sequence[i] += 1
        elif d_sequence[i] > 2: d_sequence[i] -= 1
            
    final_h = sum(math.log2(d) for d in d_sequence)
    final_t = 2 * sum(d_sequence)
    return final_h, final_t, L, d_sequence

def generate_logic_params_normal(mu_h, sigma_h, mu_t, sigma_t):
    """训练集逻辑：H, T, L 全部服从正态分布"""
    target_h = max(2.0, np.random.normal(mu_h, sigma_h))
    target_t = max(target_h * 3.0, np.random.normal(mu_t, sigma_t))
    target_m = int(target_t // 2) 
    
    ratio = target_m / target_h
    best_d_avg = 2.0
    min_diff = float('inf')
    for d_test in np.linspace(2.0, 10.0, 100):
        curr_ratio = d_test / math.log2(d_test)
        if abs(curr_ratio - ratio) < min_diff:
            min_diff = abs(curr_ratio - ratio); best_d_avg = d_test
            
    L = max(2, int(round(target_m / best_d_avg)))
    d_sequence = [max(2, target_m // L)] * L
    
    diff_m = target_m - sum(d_sequence)
    for _ in range(abs(diff_m)):
        idx = random.randint(0, L - 1)
        if diff_m > 0: d_sequence[idx] += 1
        elif d_sequence[idx] > 2: d_sequence[idx] -= 1
    return sum(math.log2(d) for d in d_sequence), 2 * sum(d_sequence), L, d_sequence

def calculate_wl_hash(edges):
    """使用 Weisfeiler-Lehman 图哈希进行拓扑唯一性验证"""
    G = nx.DiGraph(); G.add_edges_from(edges)
    return nx.weisfeiler_lehman_graph_hash(G)

def construct_and_map(h_params, all_token_ids):
    """将抽象逻辑图映射到具体的 Token ID 序列"""
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
    
    return {
        "h": round(final_h, 2), "t": final_t, "l": L, 
        "d_seq": str(d_seq), "q": q_content, "a": a_content, "hash": topo_hash
    }

# ==========================================
# 2. 存储与 Worker 逻辑
# ==========================================

def worker_task(args):
    mode, l_val, mu_h, sigma_h, mu_t, sigma_t, all_token_ids = args
    if mode == "fixed":
        params = generate_logic_params_fixed_l(l_val, mu_h, sigma_h, mu_t, sigma_t)
    else:
        params = generate_logic_params_normal(mu_h, sigma_h, mu_t, sigma_t)
    return construct_and_map(params, all_token_ids)

def save_metadata_and_stats(results, base_path, is_test=False):
    """保存元数据到 CSV，方便后续绘图和审计"""
    csv_dir = os.path.join(base_path, "metadata")
    os.makedirs(csv_dir, exist_ok=True)
    opened_files = {}
    for item in results:
        group_key = f"L{item['l']}" if is_test else f"H{int(item['h'])}"
        if group_key not in opened_files:
            p = os.path.join(csv_dir, f"{group_key}.csv")
            is_new = not os.path.exists(p)
            f = open(p, "a", newline='')
            writer = csv.writer(f)
            if is_new: writer.writerow(["h", "t", "l", "d_sequence", "hash"])
            opened_files[group_key] = (f, writer)
        opened_files[group_key][1].writerow([item['h'], item['t'], item['l'], item['d_seq'], item['hash']])
    for f, _ in opened_files.values(): f.close()

# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    # --- 配置参数 ---
    ROOT_PATH = "data/ECT-Logic"
    TRAIN_PATH = os.path.join(ROOT_PATH, "train")
    TEST_PATH = os.path.join(ROOT_PATH, "test")
    
    TOTAL_TRAIN = 1000000 
    BATCH_SIZE = 5000       
    SAVE_BLOCK_SIZE = 100000
    NUM_CORES = 32

    SAMPLES_PER_L = 1000        # 测试集每个桶的数量
    TEST_L_RANGE = range(2, 26) # 测试集覆盖的 L 范围
    
    all_token_ids = list(range(1, 2048)) # 0 留空给 Padding
    
    manager = Manager()
    global_hashes = manager.dict() # 全局哈希表，确保训练集与测试集不重叠
    
    import datasets
    datasets.disable_progress_bar()

    # ------------------------------------------
    # 阶段 1：生成训练集
    # ------------------------------------------
    print(f"\n>>> 阶段 1: 生成训练集 [目标: {TOTAL_TRAIN}]")
    os.makedirs(TRAIN_PATH, exist_ok=True)
    train_buffer = {"q": [], "a": [], "h": [], "t": [], "l": []}
    train_count = 0
    shard_idx = 0

    with Pool(NUM_CORES) as pool:
        while train_count < TOTAL_TRAIN:
            args = ("normal", None, 15, 5, 80, 20, all_token_ids)
            batch_raw = pool.map(worker_task, [args] * BATCH_SIZE)
            
            unique_batch = []
            for item in batch_raw:
                if item['hash'] not in global_hashes:
                    global_hashes[item['hash']] = True
                    unique_batch.append(item)
                    for k in train_buffer.keys(): train_buffer[k].append(item[k])
            
            save_metadata_and_stats(unique_batch, TRAIN_PATH, is_test=False)
            train_count += len(unique_batch)
            eff = (len(unique_batch)/BATCH_SIZE)*100
            print(f"训练进度: {train_count}/{TOTAL_TRAIN} | 成功率: {eff:.1f}%")

            if len(train_buffer["q"]) >= SAVE_BLOCK_SIZE:
                Dataset.from_dict(train_buffer).save_to_disk(os.path.join(TRAIN_PATH, "hf_shards", f"shard_{shard_idx}"))
                train_buffer = {k: [] for k in train_buffer.keys()}
                shard_idx += 1
        
        if train_buffer["q"]:
            Dataset.from_dict(train_buffer).save_to_disk(os.path.join(TRAIN_PATH, "hf_shards", f"shard_final"))

    # ------------------------------------------
    # 阶段 2：生成测试集 (分桶 L 采样)
    # ------------------------------------------
    print(f"\n>>> 阶段 2: 生成测试集 [L: {TEST_L_RANGE.start}-{TEST_L_RANGE.stop-1}]")
    os.makedirs(TEST_PATH, exist_ok=True)
    
    with Pool(NUM_CORES) as pool:
        for l_val in TEST_L_RANGE:
            test_buffer = {"q": [], "a": [], "h": [], "t": [], "l": []}
            l_generated = 0
            attempts = 0 # 用于计算成功率
            
            while l_generated < SAMPLES_PER_L:
                batch_needed = (SAMPLES_PER_L - l_generated) + 100
                args = ("fixed", l_val, 15, 5, 80, 20, all_token_ids)
                batch_raw = pool.map(worker_task, [args] * batch_needed)
                attempts += batch_needed
                
                unique_batch = []
                for item in batch_raw:
                    if l_generated < SAMPLES_PER_L and item['hash'] not in global_hashes:
                        global_hashes[item['hash']] = True
                        unique_batch.append(item)
                        for k in test_buffer.keys(): test_buffer[k].append(item[k])
                        l_generated += 1
                
                save_metadata_and_stats(unique_batch, TEST_PATH, is_test=True)
            
            eff = (l_generated / attempts) * 100
            print(f"测试桶 L={l_val:02d} 已完成 | 总成功率: {eff:.1f}%")
            Dataset.from_dict(test_buffer).save_to_disk(os.path.join(TEST_PATH, "hf_shards", f"shard_L{l_val}"))

    # ------------------------------------------
    # 阶段 3：绘图分析
    # ------------------------------------------
    print("\n>>> 阶段 3: 执行全量绘图...")

    # 1. 训练集统计图
    csv_train = glob.glob(os.path.join(TRAIN_PATH, "metadata/H*.csv"))
    if csv_train:
        th, tt, tl = [], [], []
        for f in csv_train:
            with open(f, 'r') as fi:
                reader = csv.DictReader(fi)
                for r in reader: th.append(float(r['h'])); tt.append(int(r['t'])); tl.append(int(r['l']))
        
        plt.figure(figsize=(18, 5))
        for i, (data, title, c) in enumerate([(th, 'H (Entropy)', '#2ecc71'), (tt, 'T (Tokens)', '#3498db'), (tl, 'L (Depth)', '#f39c12')]):
            plt.subplot(1, 3, i+1)
            plt.hist(data, bins=50 if i<2 else range(min(data), max(data)+2), color=c, alpha=0.7, edgecolor='black', density=i<2)
            if i < 2:
                mu, std = norm.fit(data); x = np.linspace(min(data), max(data), 100); plt.plot(x, norm.pdf(x, mu, std), 'r--')
            plt.title(f'Train {title}')
        plt.tight_layout(); plt.savefig(os.path.join(TRAIN_PATH, "train_summary.png"))

    # 2. 测试集统计图 (H & T 对称展示)
    csv_tests = glob.glob(os.path.join(TEST_PATH, "metadata/L*.csv"))
    if csv_tests:
        test_plot_dir = os.path.join(TEST_PATH, "plots"); os.makedirs(test_plot_dir, exist_ok=True)
        for f_path in csv_tests:
            l_label = os.path.basename(f_path).replace(".csv", "")
            h_v, t_v = [], []
            with open(f_path, 'r') as fi:
                reader = csv.DictReader(fi)
                for r in reader: h_v.append(float(r['h'])); t_v.append(int(r['t']))
            
            plt.figure(figsize=(12, 5))
            # H 子图
            plt.subplot(1, 2, 1); plt.hist(h_v, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black', density=True)
            plt.title(f'{l_label} - Entropy (H)'); plt.xlabel('Bits')
            # T 子图
            plt.subplot(1, 2, 2); plt.hist(t_v, bins=30, color='#e67e22', alpha=0.7, edgecolor='black', density=True)
            plt.title(f'{l_label} - Length (T)'); plt.xlabel('Tokens')
            
            plt.tight_layout(); plt.savefig(os.path.join(test_plot_dir, f"{l_label}_stats.png")); plt.close()
        print(f"测试集统计图已保存至 {test_plot_dir}")

    print("\n>>> 任务结束。训练与测试数据已物理隔离，成功率已在日志输出。")