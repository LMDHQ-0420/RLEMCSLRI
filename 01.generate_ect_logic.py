import networkx as nx
import random
import math
import hashlib
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import os
import gc
from collections import Counter
from multiprocessing import Pool

# ---------------- 探测与生成逻辑 ----------------

def get_natural_stats(num_edges, num_pre_samples=100):
    max_entropies = []
    path_lengths = []
    num_graph_nodes = max(5, int(num_edges * 0.3)) 
    
    for _ in range(num_pre_samples):
        temp_G = nx.DiGraph()
        nodes = list(range(num_graph_nodes))
        temp_G.add_nodes_from(nodes)
        edges = []
        while len(edges) < num_edges:
            u, v = random.sample(nodes, 2)
            if (u, v) not in edges:
                edges.append((u, v))
                temp_G.add_edge(u, v)
        
        out_degrees = dict(temp_G.out_degree())
        graph_max_h = 0
        starts = [n for n in nodes if out_degrees.get(n, 0) > 0]
        for start_node in starts:
            queue = [(start_node, {start_node}, 0.0)]
            while queue:
                curr, path, curr_h = queue.pop(0)
                graph_max_h = max(graph_max_h, curr_h)
                b = out_degrees.get(curr, 0)
                step_h = math.log2(b) if b > 1 else 0.0
                if len(path) < 12 and b > 0: 
                    for nbr in temp_G.neighbors(curr):
                        if nbr not in path:
                            new_path = path.copy()
                            new_path.add(nbr)
                            queue.append((nbr, new_path, curr_h + step_h))
                            path_lengths.append(len(new_path))
            if len(queue) > 300: break 
        max_entropies.append(graph_max_h)
    
    mode_len = Counter(path_lengths).most_common(1)[0][0] if path_lengths else 5
    return float(round(np.mean(max_entropies))), mode_len

def generate_single_sample(vocab_size, num_edges, target_H):
    all_vocab_nodes = list(range(1, vocab_size))
    num_graph_nodes = max(5, int(num_edges * 0.3))
    while True:
        temp_G = nx.DiGraph()
        temp_G.add_nodes_from(range(num_graph_nodes))
        edges = []
        while len(edges) < num_edges:
            u, v = random.sample(range(num_graph_nodes), 2)
            if (u, v) not in edges:
                edges.append((u, v))
                temp_G.add_edge(u, v)
        out_degrees = dict(temp_G.out_degree())
        possible_starts = [n for n in temp_G.nodes() if out_degrees.get(n, 0) > 0]
        if not possible_starts: continue
        random.shuffle(possible_starts)
        for start_node in possible_starts:
            queue = [(start_node, [start_node], 0.0)]
            while queue:
                curr_node, path, curr_h = queue.pop(0)
                if target_H - 0.2 <= curr_h <= target_H + 0.2:
                    involved = list(set([n for e in edges for n in e] + path))
                    token_map = {old: new for old, new in zip(involved, random.sample(all_vocab_nodes, len(involved)))}
                    q_flat = []
                    for u, v in edges:
                        q_flat.extend([token_map[u], token_map[v]])
                    q_seq = q_flat + [token_map[path[0]], token_map[path[-1]]]
                    a_seq = [token_map[n] for n in path]
                    return q_seq, a_seq, curr_h
                if curr_h < target_H + 0.5:
                    b = out_degrees.get(curr_node, 0)
                    if b > 0:
                        step_h = math.log2(b) if b > 1 else 0.0
                        for nbr in temp_G.neighbors(curr_node):
                            if nbr not in path:
                                queue.append((nbr, path + [nbr], curr_h + step_h))

def generate_random_sample(vocab_size, q_len, a_len):
    all_tokens = list(range(1, vocab_size))
    q_seq = random.choices(all_tokens, k=q_len)
    a_seq = random.choices(all_tokens, k=a_len)
    return q_seq, a_seq, 0.0

# ---------------- 处理与落盘逻辑 ----------------

def process_and_save(q_len, train_num, test_num, vocab_size, base_root, mode="logic", chunk_size=100000):
    num_edges = (q_len - 2) // 2
    target_h, mode_a_len = get_natural_stats(num_edges)
    
    data_type = "ECT-Logic" if mode == "logic" else "ECT-Random"
    current_dir = f"{base_root}/{data_type}/Q_LEN_{q_len}_H_{int(target_h)}"
    
    # 路径检查：如果已存在则跳过
    if os.path.exists(current_dir):
        # 使用 print 而非 tqdm.write，因为子进程中 print 更直接
        print(f"[SKIP] {mode.upper()} Q_LEN:{q_len} already exists.")
        return

    seen_hashes = set()
    splits = {"train": train_num, "test": test_num}
    
    for split_name, total_count in splits.items():
        save_path = os.path.join(current_dir, split_name)
        os.makedirs(save_path, exist_ok=True)
        
        collected = 0
        chunk_data = []
        part_idx = 1
        
        while collected < total_count:
            if mode == "logic":
                q_seq, a_seq, actual_h = generate_single_sample(vocab_size, num_edges, target_h)
            else:
                q_seq, a_seq, actual_h = generate_random_sample(vocab_size, q_len, mode_a_len)
            
            q_hash = hashlib.md5(np.array(q_seq, dtype=np.int32).tobytes()).digest()
            if q_hash not in seen_hashes:
                seen_hashes.add(q_hash)
                chunk_data.append({"question": q_seq, "answer": a_seq, "h": actual_h})
                collected += 1
                
                # 到达 Chunk 大小，执行落盘
                if len(chunk_data) >= chunk_size or collected == total_count:
                    file_path = os.path.join(save_path, f"part_{part_idx}.parquet")
                    Dataset.from_list(chunk_data).to_parquet(file_path)
                    
                    # 优化后的控制台输出：仅在落盘时打印
                    print(f"[SAVE] Mode: {mode.upper()} | Q_LEN: {q_len} | Split: {split_name} | Part: {part_idx} -> {file_path}")
                    
                    chunk_data = []
                    gc.collect()
                    part_idx += 1
    seen_hashes.clear()
# ---------------- 任务包装器 ----------------

def worker_task(args):
    ql, train_num, test_num, v_size, root_dir = args
    real_ql = ql + 2
    
    # 1. 进程启动提示 (增加颜色或明显标识)
    print(f"\n[PROCESS START] Handling Q_LEN: {real_ql} | PID: {os.getpid()}")
    
    # 2. 执行逻辑版生成
    process_and_save(real_ql, train_num, test_num, v_size, root_dir, mode="logic")
    
    # 3. 执行随机版生成
    process_and_save(real_ql, train_num, test_num, v_size, root_dir, mode="random")
    
    # 4. 进程结束提示
    print(f"[PROCESS FINISH] Completed Q_LEN: {real_ql}\n")

# ---------------- 主程序优化 ----------------

if __name__ == "__main__":
    V_SIZE = 2048
    # 保持 Q_LEN_LIST 不变
    Q_LEN_LIST = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    TRAIN_COUNT = 1000000
    TEST_COUNT = 50000
    ROOT_DIR = "data"
    NUM_PROCESSES = len(Q_LEN_LIST)

    task_args = [(ql, TRAIN_COUNT, TEST_COUNT, V_SIZE, ROOT_DIR) for ql in Q_LEN_LIST]

    print("=" * 80)
    print(f"Parallel Logic Data Generation System")
    print(f"Processes: {NUM_PROCESSES} | Total Tasks: {len(task_args)}")
    print(f"Root Directory: {ROOT_DIR}")
    print("=" * 80)
    
    # 使用 imap_unordered 提高调度效率
    with Pool(processes=NUM_PROCESSES) as pool:
        # 主进度条
        progress_bar = tqdm(total=len(task_args), desc="Overall Task Progress", unit="task")
        
        for _ in pool.imap_unordered(worker_task, task_args):
            progress_bar.update(1)
            
        progress_bar.close()

    print("=" * 80)
    print("[SUCCESS] All logic and random data generation completed.")
    print("=" * 80)