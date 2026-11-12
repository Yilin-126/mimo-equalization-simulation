import numpy as np
import csv
import os

def analyze_condition_numbers():
    # 固定种子，保证复现
    rng = np.random.default_rng(20261112) 
    N = 20000
    nt, nr = 2, 2
    
    # 1. 独立 Rayleigh 信道生成
    h_ind = (rng.standard_normal((N, nr, nt)) + 1j * rng.standard_normal((N, nr, nt))) / np.sqrt(2)
    
    # 2. 相关 Rayleigh 信道生成（接收端相关，rho=0.7）
    R = np.array([[1.0, 0.7], [0.7, 1.0]])
    w, V = np.linalg.eigh(R)
    R_sqrt = V @ np.diag(np.sqrt(w)) @ V.conj().T
    h_corr = np.array([R_sqrt @ h for h in h_ind])
    
    results = []
    
    for name, H in [("Independent", h_ind), ("Rx_Corr_0.7", h_corr)]:
        # 计算奇异值
        s = np.linalg.svd(H, compute_uv=False)
        cond = s[:, 0] / s[:, 1]         # 条件数
        inv_lmin = 1.0 / (s[:, 1] ** 2)  # ZF噪声放大因子 1/lambda_min
        
        results.append({
            "Scenario": name,
            "Mean_Cond": float(np.mean(cond)),
            "P90_Cond": float(np.percentile(cond, 90)),
            "Mean_Noise_Amp": float(np.mean(inv_lmin))
        })
        
    print("============= 2x2 信道条件数与 ZF 噪声放大 =============")
    # 手动格式化打印，代替 pandas
    print(f"{'Scenario':<15} | {'Mean_Cond':<12} | {'P90_Cond':<12} | {'Mean_Noise_Amp':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['Scenario']:<15} | {r['Mean_Cond']:<12.2f} | {r['P90_Cond']:<12.2f} | {r['Mean_Noise_Amp']:<15.4f}")
    
    # 确保目录存在并保存
    os.makedirs("results", exist_ok=True)
    csv_path = "results/condition_number_stats.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Scenario", "Mean_Cond", "P90_Cond", "Mean_Noise_Amp"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n结果已保存至: {csv_path}")

if __name__ == "__main__":
    analyze_condition_numbers()