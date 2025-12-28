# export_assets_local.py
# 作用：1. 现场训练 LCA 提取参数; 2. 搬运 TabPFN 模型
# 运行方式：双击运行，或者在终端 python export_assets_local.py

import os
import shutil
import pandas as pd
import numpy as np
import joblib
import json

# ================= 🔴 请核对你的 E 盘路径 🔴 =================
# 你的项目根目录
YOUR_PROJECT_ROOT = r"E:\code_piantoutong"

# 原始数据路径 (用于现场跑 LCA)
RAW_DATA_PATH = os.path.join(YOUR_PROJECT_ROOT, "processed_migraine_event_level_allv3.xlsx")
SHEET_NAME = "事件级_合并数据"

# TabPFN 模型目录
TABPFN_DIR = os.path.join(YOUR_PROJECT_ROOT, "WeakSupervision_TabPFN")
TABPFN_48H_DIR = os.path.join(YOUR_PROJECT_ROOT, "WeakSupervision_TabPFN_48hOnly")
CKPT_PATH = os.path.join(YOUR_PROJECT_ROOT, r"TabPFN_score\tabpfn-v2.5-regressor-v2.5_default.ckpt")

# LCA 配置 (跟你原脚本保持一致)
N_CLASSES_LCA = 6  # 你之前的结论是 K=6 最优
MAX_ITER = 300
TOL = 1e-4
ALPHA_SMOOTH = 1.0
RANDOM_STATE = 2025
# ===================================================================

# 目标输出目录 (自动创建在当前文件夹下的 models)
DEST_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(DEST_DIR, exist_ok=True)


# --- 把你的 LCA 核心算法搬过来 ---
def lca_em(X, n_classes, max_iter=300, tol=1e-4, random_state=None, alpha_smooth=1.0):
    rng = np.random.RandomState(random_state)
    N, D = X.shape
    pi = np.ones(n_classes) / n_classes
    theta = rng.uniform(0.25, 0.75, size=(n_classes, D))
    X1 = X
    X0 = 1 - X1
    prev_ll = None

    for it in range(max_iter):
        # E-step
        log_theta = np.log(theta + 1e-12)
        log_1_minus_theta = np.log(1 - theta + 1e-12)
        log_px_given_k = (X1[:, None, :] * log_theta[None, :, :] + X0[:, None, :] * log_1_minus_theta[None, :, :]).sum(
            axis=2)
        log_pi = np.log(pi + 1e-12)
        log_joint = log_px_given_k + log_pi[None, :]
        max_log = np.max(log_joint, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.sum(np.exp(log_joint - max_log), axis=1, keepdims=True) + 1e-12)
        log_gamma = log_joint - log_sum_exp
        gamma = np.exp(log_gamma)
        ll = log_sum_exp.sum()
        if prev_ll is not None and np.abs(ll - prev_ll) < tol: break
        prev_ll = ll

        # M-step
        Nk = gamma.sum(axis=0)
        pi = Nk / N
        theta = (gamma.T @ X1 + alpha_smooth) / (Nk[:, None] + 2 * alpha_smooth)

    return pi, theta


def train_and_export_lca():
    print(f"1. 正在读取原始数据: {RAW_DATA_PATH} ...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ 找不到原始数据: {RAW_DATA_PATH}")

    df = pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_NAME)

    # 过滤无效数据
    if "本次48h症状是否全部缺失" in df.columns:
        df = df[~(df["本次48h症状是否全部缺失"] == True)].copy()

    # 提取症状列
    symptom_cols = [c for c in df.columns if c.endswith("_48h")]
    print(f"   提取到 {len(symptom_cols)} 个症状特征，正在进行 LCA 训练 (K={N_CLASSES_LCA})...")

    X = df[symptom_cols].fillna(0).values.astype(int)

    # 现场训练
    pi, theta = lca_em(X, n_classes=N_CLASSES_LCA, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_STATE,
                       alpha_smooth=ALPHA_SMOOTH)

    # 保存参数
    lca_assets = {
        "pi": pi,
        "theta": theta,
        "symptom_cols": symptom_cols,
        "n_classes": N_CLASSES_LCA
    }

    save_path = os.path.join(DEST_DIR, "lca_params.pkl")
    joblib.dump(lca_assets, save_path)
    print(f"✅ LCA 训练完成，参数已保存至: {save_path}")


def copy_models():
    print("2. 正在复制 TabPFN 模型文件...")

    files_to_copy = [
        (os.path.join(TABPFN_DIR, "models", "tabpfn.pkl"), "tabpfn_longterm.pkl"),
        (os.path.join(TABPFN_48H_DIR, "models", "tabpfn_48h_only.pkl"), "tabpfn_48h_only.pkl"),
        (os.path.join(TABPFN_DIR, "models", "feat_cols.json"), "feat_cols_longterm.json"),
        (os.path.join(TABPFN_48H_DIR, "models", "feat_cols_48h_only.json"), "feat_cols_48h.json"),
        (CKPT_PATH, "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    ]

    for src, dst_name in files_to_copy:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DEST_DIR, dst_name))
            print(f"✅ 已复制: {dst_name}")
        else:
            print(f"⚠️ 警告: 文件未找到，跳过: {src}")


def main():
    try:
        train_and_export_lca()
        copy_models()
        print("\n🎉 恭喜！所有资产已准备就绪。")
        print("现在你可以运行启动脚本了。")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")


if __name__ == "__main__":
    main()
