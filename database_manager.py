# import sqlite3
# import json
# import pandas as pd
# from datetime import datetime
#
# DB_PATH = "migraine_data.db"
#
#
# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     # 简单的单表结构
#     c.execute('''CREATE TABLE IF NOT EXISTS patient_records
#                  (phone TEXT PRIMARY KEY,
#                   patient_name TEXT,
#                   age INTEGER,
#                   gender TEXT,
#                   history BOOLEAN,
#                   input_data JSON,
#                   risk_score REAL,
#                   risk_level TEXT,
#                   created_at DATETIME)''')
#     conn.commit()
#     conn.close()
#
#
# def save_record(info, data_dict, result):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#
#     phone = info['phone']
#     data_json = json.dumps(data_dict, ensure_ascii=False)
#     created_at = datetime.now()
#
#     # 核心逻辑：使用 REPLACE INTO (SQLite语法)
#     # 如果 phone 存在，则删除旧的插入新的；如果不存在，则插入。
#     # 这保证了每个人(手机号)永远只有一条最新数据。
#     c.execute("""
#         REPLACE INTO patient_records
#         (phone, patient_name, age, gender, history, input_data, risk_score, risk_level, created_at)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#     """, (phone, info['name'], info['age'], info['gender'], info['history'],
#           data_json, result['risk_prob_display'], result['risk_level'], created_at))
#
#     conn.commit()
#     conn.close()
#
#
# def get_all_data():
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql_query("SELECT * FROM patient_records", conn)
#     conn.close()
#
#     # 展平 JSON 数据以便导出分析
#     if not df.empty and 'input_data' in df.columns:
#         # 安全解析 JSON
#         json_struct = df['input_data'].apply(lambda x: json.loads(x) if x else {})
#         json_df = pd.json_normalize(json_struct)
#         # 合并
#         df = df.drop(columns=['input_data']).join(json_df)
#
#     return df


import pandas as pd
import json
from datetime import datetime
from supabase import create_client, Client
import streamlit as st


# 从 Streamlit 的云端保密区读取密码，不直接写在代码里
# 本地运行时，我们需要在 .streamlit/secrets.toml 里配置
# 但为了让你本地双击也能跑，这里加个容错
def get_db_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


def init_db():
    # 云数据库不需要本地初始化文件，直接跳过
    pass


def save_record(info, data_dict, result):
    supabase = get_db_client()
    if not supabase:
        print("❌ 数据库连接失败：未配置 Secrets")
        return

    data_payload = {
        "phone": info['phone'],
        "patient_name": info['name'],
        "age": info['age'],
        "gender": info['gender'],
        "history": info['history'],
        "input_data": data_dict,  # Supabase 会自动处理 JSON
        "risk_score": float(result['risk_prob_display']),
        "risk_level": result['risk_level'],
        "created_at": datetime.now().isoformat()
    }

    # 使用 upsert 实现 "有则更新，无则插入"
    try:
        supabase.table("patient_records").upsert(data_payload).execute()
        print("✅ 数据已同步至云端")
    except Exception as e:
        print(f"❌ 云端存储失败: {e}")


def get_all_data():
    supabase = get_db_client()
    if not supabase:
        return pd.DataFrame()

    try:
        # 获取所有数据
        response = supabase.table("patient_records").select("*").execute()
        data = response.data
        df = pd.DataFrame(data)

        # 展平 JSON
        if not df.empty and 'input_data' in df.columns:
            json_df = pd.json_normalize(df['input_data'])
            df = df.drop(columns=['input_data']).join(json_df)
        return df
    except Exception as e:
        st.error(f"读取失败: {e}")
        return pd.DataFrame()
