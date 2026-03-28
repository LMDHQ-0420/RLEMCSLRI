import networkx as nx
import random
import math
import hashlib
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import os
import shutil
import gc

def get_natural_entropy_level(num_edges, num_pre_samples=50):
    """探测当前结构下的最大逻辑熵天花板"""
    max_entropies = []
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
            if len(queue) > 300: break 
        max_entropies.append(graph_max_h)
    return float(round(np.mean(max_entropies)))

def generate_single_sample(vocab_size, num_edges, target_H):
    """生成符合目标逻辑熵的变长样本"""
    all_vocab_nodes = list(range(vocab_size))
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

def process_and_save_q_len(q_len, train_num, test_num, vocab_size, base_root, chunk_size=100000):
    """
    针对单个 Q_LEN 执行生成。
    每达到 chunk_size 条数据即写入一个 parquet 文件并清空内存。
    """
    num_edges = (q_len - 2) // 2
    target_h = get_natural_entropy_level(num_edges)
    
    current_q_dir = f"{base_root}/Q_LEN_{q_len}_H_{int(target_h)}"
    print(f"\n[任务启动] Q_LEN: {q_len} (Edges: {num_edges}) | 目标 H: {target_h}")
    
    # 记录已生成的 Question Hash，确保全局唯一（包括 Train 和 Test 之间也不重复）
    seen_hashes = set()
    splits = {"train": train_num, "test": test_num}
    
    for split_name, total_count in splits.items():
        save_path = os.path.join(current_q_dir, split_name)
        os.makedirs(save_path, exist_ok=True)
        
        pbar = tqdm(total=total_count, desc=f"生成 {split_name}")
        
        collected_in_split = 0
        chunk_data = []
        part_idx = 1
        
        while collected_in_split < total_count:
            q_seq, a_seq, actual_h = generate_single_sample(vocab_size, num_edges, target_h)
            
            # 高效去重
            q_hash = hashlib.md5(np.array(q_seq, dtype=np.int32).tobytes()).digest()
            if q_hash not in seen_hashes:
                seen_hashes.add(q_hash)
                chunk_data.append({
                    "question": q_seq,
                    "answer": a_seq,
                    "h": actual_h
                })
                collected_in_split += 1
                pbar.update(1)
                
                # 分块存储逻辑
                if len(chunk_data) >= chunk_size or collected_in_split == total_count:
                    # 转换为 Dataset 并保存
                    ds = Dataset.from_list(chunk_data)
                    file_name = os.path.join(save_path, f"part_{part_idx}.parquet")
                    ds.to_parquet(file_name)
                    
                    # 释放内存
                    chunk_data = []
                    del ds
                    gc.collect() # 显式触发垃圾回收
                    part_idx += 1
                    
        pbar.close()
    
    # 完成该 Q_LEN 后清空 Hash 表，节省内存迎接下一个 Q_LEN
    seen_hashes.clear()
    gc.collect()

if __name__ == "__main__":
    V_SIZE = 5000
    Q_LEN_LIST = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    TRAIN_COUNT = 1000000      # 100万
    TEST_COUNT = 50000        # 5万
    CHUNK_SIZE = 100000       # 每10万条存储一次

    ROOT_DIR = "data/ECT-Logic"

    for ql in Q_LEN_LIST:
        # ql + 2 传入函数，函数内部 (ql+2 - 2)//2 得到预期的边数
        process_and_save_q_len(ql + 2, TRAIN_COUNT, TEST_COUNT, V_SIZE, ROOT_DIR, chunk_size=CHUNK_SIZE)

    print("\n[所有任务完成] 数据已分块保存在 data/ECT-Logic 目录下。")