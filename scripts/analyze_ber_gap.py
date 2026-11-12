import csv
from collections import defaultdict

def analyze_ber_gap():
    filepath = "results/csi_ber_comparison.csv"
    
    # 用字典保存数据: {(snr, method): {"Perfect CSI": ber, "LS CSI": ber}}
    data = defaultdict(dict)
    
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                snr = float(row["SNR_dB"])
                method = row["Method"]
                csi = row["CSI"]
                ber = float(row["BER"])
                data[(snr, method)][csi] = ber
    except FileNotFoundError:
        print(f"找不到文件 {filepath}，请确保在项目根目录下运行此脚本。")
        return

    print("============= BER 恶化倍数分析 =============")
    
    for method in ["ZF", "MMSE"]:
        print(f"\n--- {method} 算法 ---")
        print(f"{'SNR(dB)':<8} | {'Perfect BER':<12} | {'LS BER':<12} | {'恶化倍数':<10}")
        print("-" * 50)
        
        # 按 SNR 排序
        snr_keys = sorted([k for k in data.keys() if k[1] == method], key=lambda x: x[0])
        
        for snr, _ in snr_keys:
            ber_dict = data[(snr, method)]
            if "Perfect CSI" in ber_dict and "LS CSI" in ber_dict:
                ber_p = ber_dict["Perfect CSI"]
                ber_ls = ber_dict["LS CSI"]
                
                # 防止除以 0
                ratio = ber_ls / ber_p if ber_p > 0 else float('inf')
                print(f"{snr:<8.1f} | {ber_p:<12.6f} | {ber_ls:<12.6f} | {ratio:<10.2f}x")

if __name__ == "__main__":
    analyze_ber_gap()