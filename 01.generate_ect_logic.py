'''
Entropy-Controlled Topological Logic Dataset
'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool
from transformers import GPT2Tokenizer

def generate_logic_params_minimal(mu_h, sigma_h, mu_t, sigma_t):
    # 1. 采样目标 H 和 T
    target_h = max(2.0, np.random.normal(mu_h, sigma_h))
    # 物理限制：M/H = d/log2(d)，最小比值约为 1.44 (当d=e时)
    # 为了保证生成成功率，target_t 至少应为 target_h * 2 * 1.5
    target_t = max(target_h * 3.0, np.random.normal(mu_t, sigma_t))
    
    # 2. 映射到目标边数 M
    target_m = int(target_t // 2) 
    
    # 3. 寻找平均分布参数
    ratio = target_m / target_h
    best_d_avg = 2.0
    min_diff = float('inf')
    for d_test in np.linspace(2.0, 10.0, 100):
        curr_ratio = d_test / math.log2(d_test)
        if abs(curr_ratio - ratio) < min_diff:
            min_diff = abs(curr_ratio - ratio)
            best_d_avg = d_test
            
    L = max(2, int(round(target_m / best_d_avg)))
    
    # 4. 线性对齐 M (固定 T)
    base_d = max(2, target_m // L)
    d_sequence = [base_d] * L
    diff_m = target_m - sum(d_sequence)
    for _ in range(abs(diff_m)):
        idx = random.randint(0, L - 1)
        if diff_m > 0:
            d_sequence[idx] += 1
        elif d_sequence[idx] > 2:
            d_sequence[idx] -= 1

    # --- 5. 核心微调逻辑：对数非线性微调 H ---
    # 目标：在保持 sum(d) 不变的情况下，通过调整 d 的离散度来逼近 target_h
    for _ in range(200):  # 迭代微调
        current_h = sum(math.log2(d) for d in d_sequence)
        error = target_h - current_h
        if abs(error) < 0.05: # 达到容差范围
            break
            
        # 随机选两个位置进行“能量交换”
        i, j = random.sample(range(L), 2)
        if d_sequence[i] <= 2 and d_sequence[j] <= 2: continue
        
        # 记录交换前的熵
        old_local_h = math.log2(d_sequence[i]) + math.log2(d_sequence[j])
        
        # 尝试调整：一个加1，一个减1 (总和 M 不变)
        if error > 0:
            # 需要增加 H -> 减小 d 的差异 (让分布更平均)
            # 原理：log(avg) + log(avg) > log(avg-delta) + log(avg+delta)
            if abs(d_sequence[i] - d_sequence[j]) > 1:
                step = 1 if d_sequence[i] < d_sequence[j] else -1
                d_sequence[i] += step
                d_sequence[j] -= step
        else:
            # 需要减小 H -> 增大 d 的差异 (让分布更极端)
            # 随机让一个变大，一个变小
            idx_sub = i if d_sequence[i] > 2 else j
            idx_add = j if idx_sub == i else i
            d_sequence[idx_sub] -= 1
            d_sequence[idx_add] += 1
            
    final_h = sum(math.log2(d) for d in d_sequence)
    final_m = sum(d_sequence)
    final_t = 2 * final_m
    
    return final_h, final_t, L, d_sequence





def worker(_):
    # 封装给进程池调用
    return generate_logic_params_minimal(15, 5, 80, 20)

if __name__ == "__main__":
    # 1. 初始化 Tokenizer（第一次运行会自动从云端下载）
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. 获取完整的词典
    # vocab 是一个字典，格式为 { "token_string": token_id }
    vocab = tokenizer.get_vocab()

    # 3. 验证词典大小 (GPT-2 标准大小为 50257)
    print(f"词典总大小: {len(vocab)}")

    # 4. 获取所有 Token ID 的列表（供你的 DAG 节点采样使用）
    all_token_ids = list(vocab.values())

    # 测试：查看前 10 个 ID
    print(f"前 10 个 Token ID: {all_token_ids[:10]}")




    TOTAL_SAMPLES = 1000000  # 生成1万组数据进行统计验证
    NUM_CORES = 32         # 充分利用你的多核优势
    
    print(f"正在使用 {NUM_CORES} 核并行生成 {TOTAL_SAMPLES} 组数据...")
    
    with Pool(NUM_CORES) as p:
        results = p.map(worker, range(TOTAL_SAMPLES))
    
    h_list = [r[0] for r in results]
    t_list = [r[1] for r in results]
    l_list = [r[2] for r in results]
    
    # --- 统计绘图 ---
    plt.figure(figsize=(15, 5))
    
    # 1. 绘制 H 的分布
    plt.subplot(1, 3, 1)
    plt.hist(h_list, bins=40, density=True, alpha=0.6, color='g', edgecolor='black')
    # 叠加拟合曲线
    mu_h, std_h = norm.fit(h_list)
    x = np.linspace(min(h_list), max(h_list), 100)
    plt.plot(x, norm.pdf(x, mu_h, std_h), 'r-', lw=2)
    plt.title(f'H (Entropy) Distribution\n$\mu={mu_h:.2f}, \sigma={std_h:.2f}$')
    plt.grid(axis='y', alpha=0.3)

    # 2. 绘制 T 的分布
    plt.subplot(1, 3, 2)
    plt.hist(t_list, bins=40, density=True, alpha=0.6, color='b', edgecolor='black')
    # 叠加拟合曲线
    mu_t, std_t = norm.fit(t_list)
    x = np.linspace(min(t_list), max(t_list), 100)
    plt.plot(x, norm.pdf(x, mu_t, std_t), 'r-', lw=2)
    plt.title(f'T (Tokens) Distribution\n$\mu={mu_t:.2f}, \sigma={std_t:.2f}$')
    plt.grid(axis='y', alpha=0.3)

    # 3. 绘制 L 的分布 (观察深度分布情况)
    plt.subplot(1, 3, 3)
    plt.hist(l_list, bins=range(min(l_list), max(l_list) + 2), alpha=0.7, color='orange', edgecolor='black', align='left')
    plt.title(f'L (Depth) Distribution\nRange: {min(l_list)}-{max(l_list)}')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/RLEMCSLRI/distribution_check.png')
    print(f"\n统计图表已保存至: /root/autodl-tmp/RLEMCSLRI/distribution_check.png")
    
    # 验证解耦度
    correlation = np.corrcoef(h_list, t_list)[0, 1]
    print(f"H 与 T 的相关系数: {correlation:.4f} (越接近0说明解耦越成功)")