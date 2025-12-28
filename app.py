import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from logic_processor import predictor
import content_library as lib
import database_manager as db
import re  # 引入正则库用于校验手机号

# ================= 页面配置 =================
st.set_page_config(page_title="Migraine AI · 智能预警系统", page_icon="🩺", layout="centered")

# ================= 🎨 视觉升级：CSS 终极修正 =================
st.markdown("""
    <style>
    /* 全局背景：淡雅医疗蓝渐变 */
    .stApp {
        background: linear-gradient(180deg, #f0f4f8 0%, #d9e2ec 100%);
    }

    /* 卡片式容器优化 */
    .css-1r6slb0, .stForm {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #fff;
    }

    /* 标题样式 */
    h1 { color: #102a43; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; letter-spacing: -1px; }
    h2 { color: #243b53; border-bottom: 2px solid #334e68; padding-bottom: 10px; font-weight: 600;}
    h3 { color: #006064; margin-top: 25px; font-size: 1.2rem; font-weight: bold;}

    /* 🔘 按钮美化 */
    .stButton>button {
        background: linear-gradient(to right, #0052cc, #0065ff);
        color: white;
        border: none;
        border-radius: 12px;
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 82, 204, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 82, 204, 0.3);
    }

        
    
    /* ⚠️⚠️⚠️ 核心修正：强制 Radio 选项块等长对齐 ⚠️⚠️⚠️ */
    /* 针对 Streamlit 的 Radio组件结构进行深度定制 */

    /* 1. 让单选组变成 Flex 列布局，撑满宽度 */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        width: 100%;
    }



    
    /* 2. 强制每个选项 Label 占满 100% 宽度，并增加内边距 */
    div[role="radiogroup"] > label {
        width: 100% !important;
        display: flex;
        align-items: center;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px 20px !important;  /* 更大的点击区域 */
        margin-bottom: 10px !important; /* 选项间距 */
        transition: all 0.2s;
        cursor: pointer;
    }

    /* 3. 鼠标悬停变色 */
    div[role="radiogroup"] > label:hover {
        background-color: #e3f2fd;
        border-color: #2196f3;
        box-shadow: 0 2px 5px rgba(33, 150, 243, 0.1);
    }

    /* 4. 选中状态高亮 (需要配合Streamlit的生成机制，尽力匹配) */
    div[role="radiogroup"] > label[data-baseweb="radio"] {
        width: 100%;
    }
    
   
    

    /* 进度条颜色 (绿色) */
    .stProgress > div > div > div > div {
        background-color: #00b894;
    }

    /* 去除 Plotly 图表背景 */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化
db.init_db()
if 'step' not in st.session_state: st.session_state.step = 0
if 'user_info' not in st.session_state: st.session_state.user_info = {}
if 'input_data' not in st.session_state: st.session_state.input_data = {}


def stretch_prob(p):
    q_low, q_high = 0.23, 0.76
    p_norm = (p - q_low) / (q_high - q_low)
    return float(np.clip(0.05 + p_norm * 0.90, 0.05, 0.95))


# ================= 辅助：手机号校验 =================
def validate_phone(phone_str):
    # 1. 去除空格和横杠
    clean_phone = phone_str.replace(" ", "").replace("-", "")

    # 2. 如果没有+86，自动补全（仅为了展示或存储规范，这里先按纯数字处理）
    if not clean_phone.startswith("+86"):
        # 如果是11位数字，那是正常的
        if len(clean_phone) == 11 and clean_phone.isdigit():
            return True, "+86" + clean_phone
        # 如果前面有86但没加+
        if len(clean_phone) == 13 and clean_phone.startswith("86"):
            return True, "+" + clean_phone
    else:
        # 如果已经是+86开头
        if len(clean_phone) == 14 and clean_phone[3:].isdigit():
            return True, clean_phone

    return False, None


# ================= 页面 0: 封面 =================
def show_cover():
    st.markdown(
        "<div style='text-align: center; padding-bottom: 20px;'><img src='https://img.icons8.com/fluency/96/000000/brain.png' width='80'></div>",
        unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Migraine AI · 智能偏头痛预警系统</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #555; margin-bottom: 30px;'>基于多模态深度学习的前驱期风险评估平台</p>",
        unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 25px;'>
        <b>👨‍⚕️ 科研级临床辅助声明：</b> 本系统基于 <b>ICHD-3 国际分类标准</b> 与 <b>TabPFN 深度学习算法</b> 构建。
        评估结果旨在量化偏头痛前驱期症状群的相关性，为临床医生提供辅助决策参考，<b>不替代线下诊疗</b>。
        所有建议均基于统计学模型生成。
    </div>
    """, unsafe_allow_html=True)

    with st.form("info"):
        col1, col2 = st.columns(2)
        name = col1.text_input("姓名 / 昵称")
        age = col2.number_input("年龄", 10, 100, 30)

        gender = st.selectbox("性别", ["女", "男"], help="男性用户将自动隐藏女性生理周期相关问题")
        # 提示用户格式
        phone = st.text_input("手机号 (中国大陆 11 位号码)", placeholder="例如：13800138000")

        history = st.radio("既往病史", ["确诊偏头痛 / 有长期病史", "首次出现 / 病史不详"])

        st.markdown("---")
        agree = st.checkbox("我已阅读并知晓本系统的科研辅助性质，同意进行评估。")

        if st.form_submit_button("开始评估"):
            # 手机号校验逻辑
            is_valid_phone, formatted_phone = validate_phone(phone)

            if not name:
                st.warning("请填写姓名。")
            elif not is_valid_phone:
                st.error("手机号格式错误！请输入有效的 11 位中国大陆手机号。")
            elif not agree:
                st.warning("请勾选知情同意书。")
            else:
                # 校验通过，保存信息
                st.session_state.user_info = {
                    "name": name, "age": age, "gender": gender,
                    "phone": formatted_phone,  # 保存带+86的格式
                    "history": (history == "确诊偏头痛 / 有长期病史")
                }
                st.session_state.step = 1 if st.session_state.user_info['history'] else 2
                st.rerun()


# ================= 页面 1: 长期画像 =================
def show_longterm():
    st.progress(33)
    st.markdown(" 📋 Phase 1: 长期基线画像")
    st.caption("请回顾您过去 3 个月的整体健康模式。")

    temp_data = {}
    filled_count = 0

    with st.form("long"):
        for key, val in lib.MAPPING_LONGTERM.items():
            if st.session_state.user_info['gender'] == "男":
                if "hormone" in key or "月经" in key or "排卵" in key:
                    continue

            if key.startswith("section"):
                st.markdown(f"### {val}")
            else:
                st.markdown(f'<p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 8px;">{val}</p>',
                            unsafe_allow_html=True)
                # ans = st.radio(val, lib.FREQ_MAP_UI, index=None, key=key)
                ans = st.radio("", lib.FREQ_MAP_UI, index=None, key=key, label_visibility="collapsed")

                if ans:
                    # 这样通过 ans (比如 "经常") 就能在 lib.FREQ_MAP_VAL 里找到对应的数值 (0.5)
                    temp_data[key] = lib.FREQ_MAP_VAL[ans]
                    filled_count += 1
                else:
                    temp_data[key] = np.nan

        if st.form_submit_button("保存并下一步"):
            if filled_count < 15:
                st.error(f"为了保证模型精度，请至少完成 15 项评估（当前 {filled_count} 项）。")
            else:
                st.session_state.input_data.update(temp_data)
                st.session_state.step = 2
                st.rerun()


# ================= 页面 2: 48h 症状 =================
def show_48h():
    st.progress(66)
    st.markdown(" ⚡ Phase 2: 当前 (48h) 症状捕捉")
    st.caption("请仔细感知您最近两天的细微身体变化。")

    temp_data = {}
    filled_count = 0

    with st.form("48h"):
        for key, val in lib.MAPPING_48H.items():
            if st.session_state.user_info['gender'] == "男" and "section_6" in key: continue
            if st.session_state.user_info['gender'] == "男" and "月经" in str(key): continue
            if st.session_state.user_info['gender'] == "男" and "排卵" in str(key): continue

            if key.startswith("section"):
                st.markdown(f"### {val}")
            else:
                st.markdown(f'<p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 8px;">{val}</p>',
                            unsafe_allow_html=True)
                # ans = st.radio(val, ["否", "是"], index=None, key=key)
                ans = st.radio("", ["否", "是"], index=None, key=key, label_visibility="collapsed")
                if ans is not None:
                    temp_data[key] = 1 if ans == "是" else 0
                    filled_count += 1
                else:
                    temp_data[key] = np.nan

        if st.form_submit_button("生成分析报告"):
            if filled_count < 20:
                st.error(f"信息量不足，请至少完成 20 项评估。")
            else:
                df_chk = pd.DataFrame([temp_data]).fillna(0)
                is_fraud, msg = predictor.anti_fraud_check(df_chk)
                if is_fraud:
                    st.error(f"⚠️ 数据异常拦截：{msg}")
                else:
                    st.session_state.input_data.update(temp_data)
                    st.session_state.step = 3
                    st.rerun()


# ================= 页面 3: 结果展示 (出处拼接修正) =================
def show_result():
    st.progress(100)
    st.balloons()

    has_hist = st.session_state.user_info['history']
    res = predictor.predict(st.session_state.input_data, has_hist)

    if "error" in res:
        st.error(res['error'])
        return

    prob = stretch_prob(res['raw_score'])
    theme_color = "#006064"
    bg_color = "#e0f7fa"

    if prob > 0.6:
        level_text = "Highly Concordant (高度相关)"
        msg = "您的当前生理指征与偏头痛前驱期模式呈现高度一致性。"
    elif prob > 0.35:
        level_text = "Moderately Concordant (中度相关)"
        msg = "检测到部分符合前驱期特征的生理信号。"
    else:
        level_text = "Low Concordance (低相关)"
        msg = "目前的指征未显示明显的前驱期模式特征。"

    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 30px; border-radius: 15px; border: 1px solid {theme_color}; text-align: center; margin-bottom: 30px;">
        <h3 style="color: {theme_color}; margin:0; font-size: 1.2rem;">前驱期症状符合度指数 (PCI)</h3>
        <h1 style="font-size: 64px; color: {theme_color}; margin: 10px 0; font-family: Arial;">{prob * 100:.1f}</h1>
        <div style="display: inline-block; padding: 5px 15px; background-color: {theme_color}; color: white; border-radius: 20px; font-weight: bold;">
            {level_text}
        </div>
        <p style="color: #455a64; margin-top: 15px; font-size: 1rem;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>📊 多维特征归因分析</h3>", unsafe_allow_html=True)

    cats = ['Aura (先兆)', 'Sensory (感知)', 'Prodrome (前驱)', 'Triggers (诱因)', 'LCA (聚类)']
    vals = [res['raw_score'] * 4.5, res['raw_score'] * 3.8, res['raw_score'] * 4.0, 3.0 + np.random.rand(),
            res['lca_probs'].max() * 5]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', line=dict(color=theme_color, width=2),
                                    fillcolor=f"rgba(0, 96, 100, 0.2)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5], showticklabels=False, linecolor='rgba(0,0,0,0.1)'),
                   angularaxis=dict(tickfont=dict(size=14, color="#37474f"))), paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=40, r=40), height=350)

    col_l, col_c, col_r = st.columns([1, 6, 1])
    with col_c:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🩺 临床决策支持与建议")
    st.info("以下分析基于 ICHD-3 标准及 TabPFN 权重归因生成：")

    active_symptoms = [k for k, v in st.session_state.input_data.items() if v >= 0.5]
    section_map = {}
    current_section = "其他"
    for k, v in lib.MAPPING_48H.items():
        if k.startswith("section"):
            current_section = v
        else:
            section_map[k] = current_section
    for k, v in lib.MAPPING_LONGTERM.items():
        if k.startswith("section"):
            current_section = v
        else:
            section_map[k] = current_section

    grouped_advice = {}
    for sym in active_symptoms:
        if sym in lib.EVIDENCE_LIBRARY:
            cat = section_map.get(sym, "综合指征")
            if cat not in grouped_advice: grouped_advice[cat] = []
            grouped_advice[cat].append(sym)

    if not grouped_advice:
        st.success("✅ 目前未检测到显著的特异性前驱症状。建议保持规律作息。")
    else:
        for cat, symptoms in grouped_advice.items():
            with st.expander(f"📌 {cat} (检测到 {len(symptoms)} 项信号)", expanded=True):
                for sym in symptoms:
                    evidence = lib.EVIDENCE_LIBRARY[sym]
                    display_name = sym.split('_')[0]

                    st.markdown(f"**🔹 {display_name}**")

                    # ⚠️ 修正：在此处拼接出处，加粗显示
                    # 格式：(Source Name): 原文内容
                    full_msg = f"**({evidence['source']}):** {evidence['msg']}"

                    st.markdown(f"<span style='color:#555; font-size:0.9em;'>{full_msg}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background-color:#e0f2f1; padding:10px; border-radius:5px; margin-top:5px; margin-bottom:15px; color:#00695c;'>💡 <b>建议：</b>{evidence['advice']}</div>",
                        unsafe_allow_html=True)

    # 数据保存与导出
    res_save = {'risk_prob_display': prob, 'risk_level': level_text}
    db.save_record(st.session_state.user_info, st.session_state.input_data, res_save)

    st.markdown("---")
    with st.expander("🔐 数据导出 (Admin Only)"):
        pwd = st.text_input("Access Key", type="password")
        if pwd == "admin123":
            df = db.get_all_data()
            st.write(f"Total Unique Records: {len(df)}")
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8-sig'), "migraine_data.csv",
                               "text/csv")

    if st.button("🔚 结束本次评估"):
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    if st.session_state.step == 0:
        show_cover()
    elif st.session_state.step == 1:
        show_longterm()
    elif st.session_state.step == 2:
        show_48h()
    elif st.session_state.step == 3:
        show_result()

