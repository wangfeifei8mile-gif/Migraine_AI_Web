# logic_processor.py
import os
import joblib
import numpy as np
import pandas as pd
import json

# 设置环境变量，告诉 TabPFN 不要去下载，直接用这里的
#os.environ["TABPFN_OFFLINE"] = "1"

# 模型文件夹相对路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class MigrainePredictor:
    def __init__(self):
        print("[System] Loading models...")
        self.lca_assets = self._load_joblib("lca_params.pkl")

        # 加载 TabPFN 及其特征列
        self.model_48h = self._load_joblib("tabpfn_48h_only.pkl")
        self.model_longterm = self._load_joblib("tabpfn_longterm.pkl")

        self.feat_cols_48h = self._load_json("feat_cols_48h.json")
        self.feat_cols_longterm = self._load_json("feat_cols_longterm.json")

        print("[System] All models loaded.")

    def _load_joblib(self, name):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            return joblib.load(path)
        raise FileNotFoundError(f"Model file missing: {path}")

    def _load_json(self, name):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def anti_fraud_check(self, df_input):
        """反作弊检测: 返回 (is_fraud, reason)"""
        vals = df_input.select_dtypes(include=[np.number]).values.flatten()
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0: return True, "数据为空"
        if np.var(vals) < 0.01: return True, "检测到所有选项填写一致，请认真填写。"
        if vals.mean() > 0.95: return True, "检测到症状勾选比例异常过高(>95%)，请确认。"

        return False, None

    def calculate_lca_posterior(self, user_df):
        """LCA 在线推理"""
        pi = self.lca_assets['pi']
        theta = self.lca_assets['theta']
        symptom_cols = self.lca_assets['symptom_cols']

        # 确保只取 symptom_cols，且顺序一致
        X = pd.DataFrame()
        for col in symptom_cols:
            if col in user_df.columns:
                X[col] = user_df[col]
            else:
                X[col] = 0  # 缺失补0
        X = X.fillna(0).values

        # EM Algorithm: E-step
        log_theta = np.log(theta + 1e-12)
        log_1_minus_theta = np.log(1 - theta + 1e-12)

        log_px_given_k = (
                X[:, None, :] * log_theta[None, :, :] +
                (1 - X)[:, None, :] * log_1_minus_theta[None, :, :]
        )
        log_px_given_k = log_px_given_k.sum(axis=2)

        log_pi = np.log(pi + 1e-12)
        log_joint = log_px_given_k + log_pi[None, :]

        max_log = np.max(log_joint, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.sum(np.exp(log_joint - max_log), axis=1, keepdims=True))
        gamma = np.exp(log_joint - log_sum_exp)

        return gamma[0]

    def predict(self, user_data_dict, has_history=False):
        df = pd.DataFrame([user_data_dict])

        # 1. 确定使用哪套特征列
        all_cols = self.feat_cols_longterm if has_history else self.feat_cols_48h

        # 2. 补全列 & Missing Mask
        for c in all_cols:
            if c not in df.columns:
                df[c] = np.nan

        # 生成 missing mask
        long_cols = [c for c in df.columns if c.endswith("_长期")]
        for c in long_cols:
            df[c + "_missingmask"] = df[c].isna().astype(float)

        df = df.fillna(0)  # TabPFN 需要 0 填充

        # 3. LCA 推理
        gamma = self.calculate_lca_posterior(df)
        lca_class_id = np.argmax(gamma)

        # 注入 LCA 特征
        for k in range(len(gamma)):
            df[f"LCA_class_prob_{k}"] = gamma[k]
            # 如果特征列里需要 One-Hot，也加上
            col_onehot = f"LCA_class_{k}"
            if col_onehot in all_cols:
                df[col_onehot] = 1 if k == lca_class_id else 0

        # 4. 按正确顺序提取特征
        try:
            X = df[all_cols].values.astype(np.float32)
        except KeyError as e:
            return {"error": f"Internal Error: Feature mismatch {e}"}

        # 5. 推理
        model = self.model_longterm if has_history else self.model_48h
        raw_score = model.predict(X)[0]
        raw_score = np.clip(raw_score, 0, 1)

        return {
            "raw_score": raw_score,
            "lca_probs": gamma,
            "lca_class": lca_class_id
        }


predictor = MigrainePredictor()
