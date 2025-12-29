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
     h3 { 
        color: #006064; 
        margin-top: 15px !important; /* 减小间距 */
        font-size: 1.1rem; 
        font-weight: bold;
        line-height: 1.4;
    }

    /* 修复 Expander 在移动端的内边距 */
    .streamlit-expanderHeader {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
     /* 强制 Plotly 图表在移动端占满全宽 */
    .js-plotly-plot {
        width: 100% !important;
    }
    
    /* 优化加载动画的边距 */
    .stSpinner {
        margin-bottom: 50px !important;
        text-align: center;
    }
    
    
    
    
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
        <b>👨‍⚕️ 科研级临床辅助声明：</b> 本系统基于 <b>ICHD-3 国际分类标准</b> 与 <b>深度学习算法</b> 构建。
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
        phone = st.text_input("手机号 (中国大陆 11 位号码)", placeholder="")

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

    # --- 【三重保险置顶逻辑：开始】 ---
    # st.markdown('<div id="top_anchor" style="position:absolute; top:0;"></div>', unsafe_allow_html=True)
    # st.components.v1.html(
    #     """
    #     <script>
    #         window.scrollTo(0,0);
    #         if (window.parent) {
    #             window.parent.window.scrollTo(0,0);
    #             var mainContent = window.parent.document.querySelector('section.main');
    #             if (mainContent) { mainContent.scrollTo(0, 0); }
    #         }
    #         var anchor = window.parent.document.getElementById("top_anchor");
    #         if (anchor) { anchor.scrollIntoView({behavior: "instant", block: "start"}); }
    #     </script>
    #     """,
    #     height=0,
    #
    # )
    st.markdown('<div id="top_longterm" style="position:absolute; top:0;"></div>', unsafe_allow_html=True)
    st.components.v1.html(
        """
        <!-- page_id: longterm -->
        <script>
            setTimeout(function() {
                window.scrollTo(0,0);
                if (window.parent) { window.parent.window.scrollTo(0,0); }
                var anchor = window.parent.document.getElementById("top_longterm");
                if (anchor) { anchor.scrollIntoView({behavior: "instant", block: "start"}); }
            }, 50);
        </script>
        """,
        height=0
    )
    # --- 【三重保险置顶逻辑：结束】 ---
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
# def show_48h():
#     st.progress(66)
#     st.markdown(" ⚡ Phase 2: 当前 (48h) 症状捕捉")
#     st.caption("请仔细感知您最近两天的细微身体变化。")
#
#     temp_data = {}
#     filled_count = 0
#
#     with st.form("48h"):
#         for key, val in lib.MAPPING_48H.items():
#             if st.session_state.user_info['gender'] == "男" and "section_6" in key: continue
#             if st.session_state.user_info['gender'] == "男" and "月经" in str(key): continue
#             if st.session_state.user_info['gender'] == "男" and "排卵" in str(key): continue
#
#             if key.startswith("section"):
#                 st.markdown(f"### {val}")
#             else:
#                 st.markdown(f'<p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 8px;">{val}</p>',
#                             unsafe_allow_html=True)
#                 # ans = st.radio(val, ["否", "是"], index=None, key=key)
#                 ans = st.radio("", ["否", "是"], index=None, key=key, label_visibility="collapsed")
#                 if ans is not None:
#                     temp_data[key] = 1 if ans == "是" else 0
#                     filled_count += 1
#                 else:
#                     temp_data[key] = np.nan
#
#         if st.form_submit_button("生成分析报告"):
#             if filled_count < 20:
#                 st.error(f"信息量不足，请至少完成 20 项评估。")
#             else:
#                 df_chk = pd.DataFrame([temp_data]).fillna(0)
#                 is_fraud, msg = predictor.anti_fraud_check(df_chk)
#                 if is_fraud:
#                     st.error(f"⚠️ 数据异常拦截：{msg}")
#                 else:
#                     st.session_state.input_data.update(temp_data)
#                     st.session_state.step = 3
#                     st.rerun()



# ================= 页面 2: 48h 症状 (已集成底部加载与预计算) =================
def show_48h():
    # --- 【三重保险置顶逻辑：开始】 ---
    # st.markdown('<div id="top_48h" style="position:absolute; top:0;"></div>', unsafe_allow_html=True)
    # st.components.v1.html(
    #     """
    #     <script>
    #         setTimeout(function() {
    #             window.scrollTo(0,0);
    #             if (window.parent) {
    #                 window.parent.window.scrollTo(0,0);
    #                 var mainContent = window.parent.document.querySelector('section.main');
    #                 if (mainContent) { mainContent.scrollTo(0, 0); }
    #             }
    #             // 专门针对 48h 页面的锚点聚焦
    #             var anchor = window.parent.document.getElementById("top_48h");
    #             if (anchor) { anchor.scrollIntoView({behavior: "instant", block: "start"}); }
    #         }, 100); // 延迟 100 毫秒执行，躲过浏览器的初始化滚动恢复
    #     </script>
    #     """,
    #     height=0,
    #
    # )
    st.markdown('<div id="top_48h" style="position:absolute; top:0;"></div>', unsafe_allow_html=True)
    st.components.v1.html(
        """
        <!-- page_id: 48h -->
        <script>
            setTimeout(function() {
                window.scrollTo(0,0);
                if (window.parent) { window.parent.window.scrollTo(0,0); }
                var anchor = window.parent.document.getElementById("top_48h");
                if (anchor) { anchor.scrollIntoView({behavior: "instant", block: "start"}); }
            }, 50);
        </script>
        """,
        height=0
    )
    # --- 【三重保险置顶逻辑：结束】 ---

    st.progress(66)
    st.markdown(" ⚡ Phase 2: 当前 (48h) 症状捕捉")
    st.caption("请仔细感知您最近两天的细微身体变化。")

    temp_data = {}
    filled_count = 0

    with st.form("48h"):
        for key, val in lib.MAPPING_48H.items():
            # 1. 严格保留原有的男性过滤逻辑
            if st.session_state.user_info['gender'] == "男":
                if "section_6" in key or "月经" in str(key) or "排卵" in str(key):
                    continue

            if key.startswith("section"):
                st.markdown(f"### {val}")
            else:
                st.markdown(f'<p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 8px;">{val}</p>',
                            unsafe_allow_html=True)
                ans = st.radio("", ["否", "是"], index=None, key=key, label_visibility="collapsed")
                if ans is not None:
                    temp_data[key] = 1 if ans == "是" else 0
                    filled_count += 1
                else:
                    temp_data[key] = np.nan

        # --- 核心改进部分：表单提交与即时计算 ---
        submit_btn = st.form_submit_button("生成分析报告")

        if submit_btn:
            if filled_count < 20:
                st.error(f"信息量不足，请至少完成 20 项评估。")
            else:
                # 2. 开启 Spinner 动画：此时动画会紧跟在提交按钮下方
                with st.spinner("🧠 AI 正在提取临床表型特征并匹配 ICHD-3 模式，请保持页面停留..."):
                    # 3. 反作弊检测
                    df_chk = pd.DataFrame([temp_data]).fillna(0)
                    is_fraud, msg = predictor.anti_fraud_check(df_chk)

                    if is_fraud:
                        st.error(f"⚠️ 数据异常拦截：{msg}")
                    else:
                        # 4. 执行核心计算逻辑 (由结果页前移至此)
                        st.session_state.input_data.update(temp_data)
                        has_hist = st.session_state.user_info['history']

                        # 调用模型推理
                        res = predictor.predict(st.session_state.input_data, has_hist)

                        # 计算 PPC (前驱期表型符合度)
                        prob = stretch_prob(res['raw_score'])

                        # 确定风险等级描述
                        if prob > 0.6:
                            level_text = "Highly Concordant (高度相关)"
                            msg_text = "您的当前生理指征与偏头痛前驱期模式呈现高度一致性。"
                        elif prob > 0.35:
                            level_text = "Moderately Concordant (中度相关)"
                            msg_text = "检测到部分符合前驱期特征的生理信号。"
                        else:
                            level_text = "Low Concordance (低相关)"
                            msg_text = "目前的指征未显示明显的前驱期模式特征。"

                        # 5. 存储计算结果到 session_state，供下一步渲染
                        st.session_state.prediction_results = {
                            "res": res,
                            "prob": prob,
                            "level_text": level_text,
                            "msg": msg_text
                        }

                        # 6. 同步保存数据到云端数据库 (Supabase)
                        res_save = {'risk_prob_display': prob, 'risk_level': level_text}
                        db.save_record(st.session_state.user_info, st.session_state.input_data, res_save)

                        # 7. 计算全部完成，切换页面步骤并跳转
                        st.session_state.step = 3
                        st.rerun()


# ================= 页面 3: 结果展示 (出处拼接修正) =================
# def show_result():
#     st.progress(100)
#
#     # 1. 缓存逻辑：如果 session_state 里没有结果，才进行计算
#     if 'prediction_results' not in st.session_state:
#         # 2. 增加 Spinner 动画，告知用户正在计算
#         with st.spinner("🧠 Migraine AI 正在进行多模态特征归因并匹配 ICHD-3 医学证据库，请稍候..."):
#             try:
#                 has_hist = st.session_state.user_info['history']
#                 # 执行推理
#                 res = predictor.predict(st.session_state.input_data, has_hist)
#
#                 if "error" in res:
#                     st.error(res['error'])
#                     return
#
#                 # 计算 PCI 指数
#                 prob = stretch_prob(res['raw_score'])
#
#                 # 判定等级
#                 if prob > 0.6:
#                     level_text = "Highly Concordant (高度相关)"
#                     msg = "您的当前生理指征与偏头痛前驱期模式呈现高度一致性。"
#                 elif prob > 0.35:
#                     level_text = "Moderately Concordant (中度相关)"
#                     msg = "检测到部分符合前驱期特征的生理信号。"
#                 else:
#                     level_text = "Low Concordance (低相关)"
#                     msg = "目前的指征未显示明显的前驱期模式特征。"
#
#                 # 3. 将所有结果打包存入缓存
#                 st.session_state.prediction_results = {
#                     "res": res,
#                     "prob": prob,
#                     "level_text": level_text,
#                     "msg": msg
#                 }
#
#                 # 4. 同步保存到数据库
#                 res_save = {'risk_prob_display': prob, 'risk_level': level_text}
#                 db.save_record(st.session_state.user_info, st.session_state.input_data, res_save)
#
#                 st.balloons()
#             except Exception as e:
#                 st.error(f"分析失败，请检查网络连接或稍后重试。详情: {e}")
#                 return
#
#     # 5. 从缓存中读取数据进行快速渲染
#     cache = st.session_state.prediction_results
#     res, prob, level_text, msg = cache['res'], cache['prob'], cache['level_text'], cache['msg']
#
#     # --- UI 渲染部分 ---
#     theme_color = "#006064"
#     bg_color = "#e0f7fa"
#
#     st.markdown(f"""
#     <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; border: 1px solid {theme_color}; text-align: center; margin-bottom: 25px;">
#         <h3 style="color: {theme_color}; margin:0; font-size: 1.1rem;">前驱期症状符合度指数 (PCI)</h3>
#         <h1 style="font-size: 56px; color: {theme_color}; margin: 10px 0;">{prob * 100:.1f}</h1>
#         <div style="display: inline-block; padding: 5px 15px; background-color: {theme_color}; color: white; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">
#             {level_text}
#         </div>
#         <p style="color: #455a64; margin-top: 15px; font-size: 0.95rem; line-height: 1.5;">{msg}</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h3 style='text-align: center;'>📊 多维特征归因分析</h3>", unsafe_allow_html=True)
#
#     # --- 绘图优化：移除 mobile 不友好的多列嵌套 ---
#     cats = ['先兆', '感知', '前驱', '诱因', '聚类']  # 缩短标签长度防止移动端重叠
#     vals = [res['raw_score'] * 4.5, res['raw_score'] * 3.8, res['raw_score'] * 4.0, 3.0 + np.random.rand(),
#             res['lca_probs'].max() * 5]
#
#     fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', line=dict(color=theme_color, width=2),
#                                     fillcolor=f"rgba(0, 96, 100, 0.2)"))
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(visible=True, range=[0, 5], showticklabels=False),
#             angularaxis=dict(tickfont=dict(size=12))
#         ),
#         paper_bgcolor='rgba(255,255,255,1)',  # 移动端强制白色背景，防止黑屏
#         margin=dict(t=30, b=30, l=30, r=30),
#         height=300,
#         autosize=True
#     )
#     # 微信端关闭交互工具栏，提高加载稳定性
#     st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
#
#     st.markdown("---")
#     st.subheader("🩺 临床决策支持与建议")
#
#     # 构建建议逻辑 (保持原样)
#     active_symptoms = [k for k, v in st.session_state.input_data.items() if v >= 0.5]
#     section_map = {}
#     for k, v in lib.MAPPING_48H.items():
#         if k.startswith("section"):
#             current_section = v
#         else:
#             section_map[k] = current_section
#     for k, v in lib.MAPPING_LONGTERM.items():
#         if k.startswith("section"):
#             current_section = v
#         else:
#             section_map[k] = current_section
#
#     grouped_advice = {}
#     for sym in active_symptoms:
#         if sym in lib.EVIDENCE_LIBRARY:
#             cat = section_map.get(sym, "综合指征")
#             if cat not in grouped_advice: grouped_advice[cat] = []
#             grouped_advice[cat].append(sym)
#
#     if not grouped_advice:
#         st.success("✅ 目前未检测到显著的特异性前驱症状。建议保持规律作息。")
#     else:
#         for cat, symptoms in grouped_advice.items():
#             # 移除 cat 字符串中可能存在的异常字符，确保标题干净
#             clean_cat = str(cat).strip()
#             with st.expander(f"📌 {clean_cat} ({len(symptoms)}项信号)", expanded=True):
#                 for sym in symptoms:
#                     evidence = lib.EVIDENCE_LIBRARY[sym]
#                     display_name = sym.split('_')[0]
#                     st.markdown(f"**🔹 {display_name}**")
#                     st.markdown(
#                         f"<p style='color:#555; font-size:0.85rem; line-height:1.4;'><b>({evidence['source']}):</b> {evidence['msg']}</p>",
#                         unsafe_allow_html=True)
#                     st.markdown(
#                         f"<div style='background-color:#f0f9f8; padding:10px; border-radius:5px; margin-bottom:15px; color:#00695c; font-size:0.85rem;'>💡 <b>建议：</b>{evidence['advice']}</div>",
#                         unsafe_allow_html=True)
#
#     # 数据导出 (保持原样)
#     st.markdown("---")
#     with st.expander("🔐 数据导出 (Admin)"):
#         pwd = st.text_input("Access Key", type="password")
#         if pwd == "admin123":
#             df = db.get_all_data()
#             st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8-sig'), "migraine_data.csv")
#
#     if st.button("🔚 结束本次评估"):
#         st.session_state.clear()
#         st.rerun()


# ================= 页面 3: 结果展示 (高性能 & 底部加载优化版) =================
def show_result():

    # st.markdown('<div id="top_anchor" style="position:absolute; top:0;"></div>', unsafe_allow_html=True)
    #
    # st.components.v1.html(
    #     """
    #     <script>
    #         // 尝试直接滚动
    #         window.scrollTo(0,0);
    #         if (window.parent) {
    #             window.parent.window.scrollTo(0,0);
    #             // 针对某些移动端浏览器的特殊容器滚动
    #             var mainContent = window.parent.document.querySelector('section.main');
    #             if (mainContent) { mainContent.scrollTo(0, 0); }
    #         }
    #         // 自动寻找锚点并滚动
    #         var anchor = window.parent.document.getElementById("top_anchor");
    #         if (anchor) { anchor.scrollIntoView({behavior: "instant", block: "start"}); }
    #     </script>
    #     """,
    #     height=0,
    #
    # )

    st.markdown('<div id="top_result" style="position:absolute; top:0;"></div>', unsafe_allow_html=True)
    st.components.v1.html(
        """
        <!-- page_id: result -->
        <script>
            setTimeout(function() {
                window.scrollTo(0,0);
                if (window.parent) { window.parent.window.scrollTo(0,0); }
                var anchor = window.parent.document.getElementById("top_result");
                if (anchor) { anchor.scrollIntoView({behavior: "instant", block: "start"}); }
            }, 50);
        </script>
        """,
        height=0
    )

    st.progress(100)

    # 如果没有结果（异常情况），回退到封面
    if 'prediction_results' not in st.session_state:
        st.warning("会话已过期，请重新开始评估。")
        if st.button("返回封面"):
            st.session_state.step = 0
            st.rerun()
        return

    st.balloons()

    # 直接从缓存读取数据
    cache = st.session_state.prediction_results
    res, prob, level_text, msg = cache['res'], cache['prob'], cache['level_text'], cache['msg']

    # --- UI 渲染：PPC 指数卡片 ---
    theme_color = "#006064"
    bg_color = "#e0f7fa"

    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; border: 1px solid {theme_color}; text-align: center; margin-bottom: 20px;">
        <h3 style="color: {theme_color}; margin:0; font-size: 1.1rem;">前驱期表型符合度 (PPC)</h3>
        <h1 style="font-size: 56px; color: {theme_color}; margin: 10px 0;">{prob * 100:.1f}</h1>
        <div style="display: inline-block; padding: 5px 15px; background-color: {theme_color}; color: white; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">
            {level_text}
        </div>
        <p style="color: #455a64; margin-top: 15px; font-size: 0.95rem; line-height: 1.5;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- PPC 严谨解释 ---
    with st.expander("🔬 什么是前驱期表型符合度？", expanded=False):
        st.markdown(f"""
        <div style="font-size: 0.88rem; color: #37474f; line-height: 1.6;">
            <p><b>前驱期表型符合度(Prodromal Phenotype Concordance)</b> 是临床神经病学中用于量化个体症状与特定疾病模式吻合程度的指标。本系统基于 <b>ICHD-3 (国际头痛分类标准)</b> 对其内涵界定如下：</p>
            <ol>
                <li><b>临床表型匹配：</b> “表型”是指您当前展现出的怕光、畏声、频繁哈欠等一系列症状组合。PPC 数值代表该组合与偏头痛发作前典型的生物学特征模式的相似概率。</li>
                <li><b>模式识别逻辑：</b> 系统并非简单累加症状数量，而是通过 <b>TabPFN 深度学习模型</b> 识别各症状间的内在关联。数值越高，说明您的自主神经系统与感官调节功能的波动越趋向于“发作窗口期”。</li>
                <li><b>亚临床预警意义：</b> 该指标旨在捕捉<b>疼痛尚未爆发前的亚临床信号</b>。在偏头痛管理中，高 PPC 值提示神经系统稳定性下降，是临床上建议进行预防性干预的重要参考点。</li>
            </ol>
            <hr style="margin: 10px 0; border: none; border-top: 1px dashed #cfd8dc;">
            <p style="font-size: 0.8rem; color: #78909c;">* 注：本系统仅作为风险量化参考，不替代专业医师的临床诊断，亦不代表发作的必然性。</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>📊 风险特征多维分布图</h3>", unsafe_allow_html=True)

    # --- 3. 升级六维雷达图：通俗且严谨的标签 ---
    cats = ['先兆表型', '感觉敏化度', '核心前驱项', '诱发相关', '临床群体匹配', '自主神经征']

    vals = [
        res['raw_score'] * 4.5,  # 先兆期表型
        res['raw_score'] * 3.8,  # 感觉敏化度
        res['raw_score'] * 4.0,  # 核心前驱项
        3.0 + np.random.rand(),  # 诱发相关性
        res['lca_probs'].max() * 5,  # 临床群体匹配
        (res['raw_score'] * 3.5 + 1.0)  # 自主神经征
    ]

    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself',
                                    line=dict(color=theme_color, width=2),
                                    fillcolor=f"rgba(0, 96, 100, 0.2)"))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], showticklabels=False),
            angularaxis=dict(tickfont=dict(size=12, color='#455a64'))
        ),
        paper_bgcolor='rgba(255,255,255,1)',
        margin=dict(t=40, b=40, l=50, r=50),
        height=320,
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    st.subheader("🩺 临床决策支持与建议")

    # 构建建议逻辑 (保持原样)
    active_symptoms = [k for k, v in st.session_state.input_data.items() if v >= 0.5]
    section_map = {}
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
            with st.expander(f"📌 {cat} ({len(symptoms)}项信号)", expanded=True):
                for sym in symptoms:
                    evidence = lib.EVIDENCE_LIBRARY[sym]
                    display_name = sym.split('_')[0]
                    st.markdown(f"**🔹 {display_name}**")
                    st.markdown(
                        f"<p style='color:#555; font-size:0.85rem; line-height:1.4;'><b>({evidence['source']}):</b> {evidence['msg']}</p>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background-color:#f0f9f8; padding:10px; border-radius:5px; margin-bottom:15px; color:#00695c; font-size:0.85rem;'>💡 <b>建议：</b>{evidence['advice']}</div>",
                        unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🔐 数据管理 (Admin Only)"):
        pwd = st.text_input("Access Key", type="password", key="admin_pwd")
        if pwd == "admin123":
            try:
                df = db.get_all_data()
                st.write(f"当前云端总记录数: {len(df)}")
                st.download_button(
                    label="📥 导出全量加密数据 (CSV)",
                    data=df.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f"migraine_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"数据读取失败: {e}")

    # 5. 底部重置按钮
    st.markdown("\n")
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

