# 导入必要的库
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 设置页面标题和布局
st.set_page_config(page_title="Cardiometabolic Multimorbidity Prediction Model", layout="wide", page_icon="🏥")

# 添加CSS样式和动画效果
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* 主卡片样式（仅用于标题） */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    /* 结果卡片样式 */
    .success-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
    }
    
    .error-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
    }
    
    .info-card {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.2);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* 功能卡片样式（仅用于欢迎页） */
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* 表格样式 */
    .feature-table {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 模型文件路径
MODEL_PATH = "LR_model.sav"

# === 连续变量归一化参数 ===
feature_mins = np.array([
    65.0,              # age_min
    35.0,              # temperature_min  
    36.0,              # pulse_min
    0.1,               # left_naked_eye_min
    0.1,               # right_naked_eye_min
    36.0,              # heart_rate_min
    50.0,              # HGB_min
    41.0,              # PLT_min
    24.0,              # Creatinine_min
    1.0,               # Urea_min
    1.0,               # TC_min
    0.33,              # LDL_C_min
    0.1326,            # HDL_C_min
    78.0,              # systolic_min
    50.0,              # diastolic_min
    -0.285714285714278, # RFM_min
    7.22076951556516,  # TyG_min
    120.347713086439,  # TyG_BMI_min
    2.40941559543402,  # TyG_WHtR_min
    1.0                # count_min
])

feature_maxs = np.array([
    112.0,            # age_max
    39.0,             # temperature_max
    160.0,            # pulse_max
    2.0,              # left_naked_eye_max
    2.0,              # right_naked_eye_max
    126.0,            # heart_rate_max
    234.0,            # HGB_max
    906.0,            # PLT_max
    567.0,            # Creatinine_max
    48.8,             # Urea_max
    11.89,            # TC_max
    7.45,             # LDL_C_max
    5.0,              # HDL_C_max
    208.0,            # systolic_max
    133.0,            # diastolic_max
    51.5384615384615, # RFM_max
    11.3181415603046, # TyG_max
    595.622915922451, # TyG_BMI_max
    8.80893171427442, # TyG_WHtR_max
    3.0               # count_max
])

# 特征名称映射（20个连续变量 + 8个分类变量）
continuous_feature_names = [
    'age','temperature','pulse','left_naked_eye','right_naked_eye','heart_rate',
    'HGB','PLT','Creatinine','Urea','TC','LDL_C','HDL_C','systolic','diastolic',
    'RFM','TyG','TyG_BMI','TyG_WHtR','count'
]

categorical_feature_names = [
    'gender', 'smoke', 'heart_rhythm', 'hear', 'exercise',
    'hypertension_final', 'diabetes_final', 'dyslipidemia_final'
]

all_feature_names = continuous_feature_names + categorical_feature_names

@st.cache_resource
def load_model():
    """
    加载LR模型
    """
    try:
        with open(MODEL_PATH, 'rb') as file:
            content = pickle.load(file)
            
        # 检查加载对象的类型
        print(f"Loaded object type: {type(content)}")
        
        # 如果是字典，提取模型
        if isinstance(content, dict) and 'model' in content:
            model = content['model']
                           
            return model, content
        else:
                       
            # 创建包含模型信息的字典
            model_content = {
                'model': content,
                'best_threshold': 0.3,  # 默认阈值
                'train_sensitivity': 0.0  # 默认灵敏度
            }
            
            return content, model_content
            
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # 显示详细错误信息
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'welcome'
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# 加载模型
model, model_content = load_model()

# 移除侧边栏阈值设置，直接在代码中设置
custom_threshold = 0.3  # 设置为固定值，不向用户显示

# 欢迎页面
def show_welcome_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 主标题卡片
    st.markdown("""
    <div class="card">
        <div style="text-align: center;">
            <h1 style="color: white; margin-bottom: 10px; font-size: 2.5em;">🏥 MyCMMrisk</h1>
            <h3 style="color: white; opacity: 0.9; margin-bottom: 20px;">Cardiometabolic Multimorbidity Risk Assessment System</h3>
            <p style="color: white; opacity: 0.8;">AI-powered risk assessment tool for healthcare professionals</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能介绍卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <h3 style="color: #667eea;">🔍</h3>
                <h4 style="color: #333;">Early Identification</h4>
                <p style="color: #666; font-size: 14px;">Identify high-risk populations</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <h3 style="color: #667eea;">📊</h3>
                <h4 style="color: #333;">Scientific Assessment</h4>
                <p style="color: #666; font-size: 14px;">Evaluate individual risk probability</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <h3 style="color: #667eea;">💡</h3>
                <h4 style="color: #333;">Decision Support</h4>
                <p style="color: #666; font-size: 14px;">Provide clinical intervention reference</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <h3 style="color: #667eea;">🎯</h3>
                <h4 style="color: #333;">Precision Prevention</h4>
                <p style="color: #666; font-size: 14px;">Enable personalized health management</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 评估流程
    st.markdown('<div class="section-header"><h3>📝 Assessment Process</h3></div>', unsafe_allow_html=True)
    
    # 步骤展示
    steps = [
        {"icon": "👤", "title": "Demographics & Lifestyle"},
        {"icon": "🔬", "title": "Physical Examination"},
        {"icon": "💉", "title": "Laboratory Indicators"},
        {"icon": "📋", "title": "Medical History"},
        {"icon": "📈", "title": "Health Risk Index"},
        {"icon": "📊", "title": "Risk Assessment Results"}
    ]
    
    cols = st.columns(6)
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {'#f0f2f6' if i < 5 else '#e8f4fd'}; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="margin: 10px 0;">{step['icon']}</h2>
                <p style="font-weight: bold; margin: 5px 0; font-size: 12px;">Step {i+1}</p>
                <p style="font-size: 11px; color: #666;">{step['title']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 目标人群
    st.markdown('<div class="section-header"><h3>🎯 Target Population</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>👥 Underlying Conditions</h4>
            <p>Hypertension, diabetes, dyslipidemia</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>👨‍👩‍👧‍👦 Family History</h4>
            <p>Cardiometabolic multimorbidity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Health Conscious</h4>
            <p>Individuals interested in health risk status</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 重要说明
    st.markdown('<div class="section-header"><h3>⚠️ Important Notes</h3></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-card">
        <ul>
            <li>This system's prediction results are for reference only and cannot replace professional medical diagnosis</li>
            <li>Please ensure the accuracy and completeness of input data</li>
            <li>Consult professional medical personnel if you have any questions</li>
            <li>Age requirement: 65 years and above</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 开始评估按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
            st.session_state.current_page = 'demographics'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 人口统计学和生活方式行为页面（第1步）
def show_demographics_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 页面标题
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);">
        <h2 style="color: white; margin: 0;">👤 Step 1: Demographics and Lifestyle Behaviors</h2>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Please fill in the patient's demographic information and lifestyle behaviors</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=65, max_value=112, value=70, key="age")
        gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female", horizontal=True, key="gender")
        
    with col2:
        smoke = st.radio("Smoking", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker", horizontal=True, key="smoke")
    
    # 保存数据到会话状态
    st.session_state.form_data.update({
        'age': age,
        'gender': gender,
        'smoke': smoke
    })
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.current_page = 'welcome'
            st.rerun()
    with col2:
        if st.button("➡️ Next: Physical Examination", type="primary", use_container_width=True):
            st.session_state.current_page = 'physical_examination'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 体格检查页面（第2步）
def show_physical_examination_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 页面标题
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);">
        <h2 style="color: white; margin: 0;">🔬 Step 2: Physical Examination</h2>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Please fill in the patient's physical examination indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("Body Temperature (℃)", min_value=35.0, max_value=39.0, value=36.5, step=0.1, key="temperature")
        pulse = st.number_input("Pulse rate (beats/min)", min_value=36, max_value=160, value=72, key="pulse")
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=78, max_value=208, value=120, key="systolic")
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=133, value=80, key="diastolic")
        height_cm = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=165.0, step=0.1, key="height_cm")
        
    with col2:
        left_naked_eye = st.number_input("Visual acuity, left eye (decimal notation: 0.1-2.0)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="left_naked_eye")
        right_naked_eye = st.number_input("Visual acuity, right eye (decimal notation: 0.1-2.0)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="right_naked_eye")
        heart_rate = st.number_input("Heart Rate (beats/min)", min_value=36, max_value=126, value=72, key="heart_rate")
        WC_cm = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=80.0, step=0.1, key="WC_cm")
        weight_kg = st.number_input("Weight (kg)", min_value=31.0, max_value=180.0, value=65.0, step=0.1, key="weight_kg")
        hear = st.radio("Hearing Ability", [0, 1], format_func=lambda x: "Normal (audible)" if x == 0 else "Abnormal (inaudible or unclear)", horizontal=True, key="hear")
        exercise = st.radio("Motor Function", [0, 1], format_func=lambda x: "Normal (can be completed smoothly)" if x == 0 else "Abnormal (cannot be completed independently)", horizontal=True, key="exercise")
        heart_rhythm = st.radio("Heart Rhythm", [0, 1], format_func=lambda x: "Regular" if x == 0 else "Irregular", horizontal=True, key="heart_rhythm")
        
    # 保存数据到会话状态
    st.session_state.form_data.update({
        'temperature': temperature,
        'pulse': pulse,
        'systolic': systolic,
        'diastolic': diastolic,
        'left_naked_eye': left_naked_eye,
        'right_naked_eye': right_naked_eye,
        'heart_rate': heart_rate,
        'height_cm': height_cm,
        'WC_cm': WC_cm,
        'weight_kg': weight_kg,
        'hear': hear,
        'exercise': exercise,
        'heart_rhythm': heart_rhythm
    })
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Previous: Demographics", use_container_width=True):
            st.session_state.current_page = 'demographics'
            st.rerun()
    with col3:
        if st.button("➡️ Next: Laboratory Indicators", type="primary", use_container_width=True):
            st.session_state.current_page = 'laboratory_indicators'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 实验室指标页面（第3步）
def show_laboratory_indicators_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 页面标题
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #cc2b5e 0%, #753a88 100%);">
        <h2 style="color: white; margin: 0;">💉 Step 3: Laboratory Indicators</h2>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Please fill in the patient's laboratory test indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        HGB = st.number_input("Hemoglobin (HGB, g/L)", min_value=50.0, max_value=234.0, value=135.0, step=1.0, key="HGB")
        PLT = st.number_input("Platelet Count (PLT, ×10⁹/L)", min_value=41.0, max_value=906.0, value=250.0, step=1.0, key="PLT")
        Creatinine = st.number_input("Creatinine (μmol/L)", min_value=24.0, max_value=567.0, value=70.0, step=1.0, key="Creatinine")
        Urea = st.number_input("Urea (mmol/L)", min_value=1.0, max_value=48.8, value=5.0, step=0.1, key="Urea")
        
    with col2:
        TC = st.number_input("Total Cholesterol (TC, mmol/L)", min_value=1.0, max_value=11.89, value=4.5, step=0.1, key="TC")
        LDL_C = st.number_input("Low-Density Lipoprotein (LDL_C, mmol/L)", min_value=0.33, max_value=7.45, value=2.5, step=0.1, key="LDL_C")
        HDL_C = st.number_input("High-Density Lipoprotein (HDL_C, mmol/L)", min_value=0.1326, max_value=5.0, value=1.2, step=0.1, key="HDL_C")
        TG = st.number_input("Triglycerides (mmol/L)", min_value=0.3, max_value=20.0, value=1.2, step=0.1, key="TG")
        glucose = st.number_input("Glucose (mmol/L)", min_value=3.0, max_value=30.0, value=5.5, step=0.1, key="glucose")
    
    # 保存数据到会话状态
    st.session_state.form_data.update({
        'HGB': HGB,
        'PLT': PLT,
        'Creatinine': Creatinine,
        'Urea': Urea,
        'TC': TC,
        'LDL_C': LDL_C,
        'HDL_C': HDL_C,
        'glucose': glucose,
        'TG': TG
    })
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Previous: Physical Examination", use_container_width=True):
            st.session_state.current_page = 'physical_examination'
            st.rerun()
    with col3:
        if st.button("➡️ Next: Medical History", type="primary", use_container_width=True):
            st.session_state.current_page = 'medical_history'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 疾病史页面（第4步）
def show_medical_history_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 页面标题
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <h2 style="color: white; margin: 0;">📋 Step 4: Medical History</h2>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Please select the patient's existing chronic disease conditions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hypertension_final = st.radio("Hypertension", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes", 
                                      horizontal=True, key="hypertension_final")
    
    with col2:
        diabetes_final = st.radio("Diabetes", [0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes", 
                                 horizontal=True, key="diabetes_final")
    
    with col3:
        dyslipidemia_final = st.radio("Dyslipidemia", [0, 1], 
                                     format_func=lambda x: "No" if x == 0 else "Yes", 
                                     horizontal=True, key="dyslipidemia_final")
    
    # 计算疾病数量
    disease_count = hypertension_final + diabetes_final + dyslipidemia_final
    
    # 疾病数量显示
    st.metric("Number of Existing Diseases (Hypertension, Diabetes, Dyslipidemia)", disease_count)
    
    # 保存数据到会话状态
    st.session_state.form_data.update({
        'hypertension_final': hypertension_final,
        'diabetes_final': diabetes_final,
        'dyslipidemia_final': dyslipidemia_final,
        'count': disease_count
    })
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Previous: Laboratory Indicators", use_container_width=True):
            st.session_state.current_page = 'laboratory_indicators'
            st.rerun()
    with col3:
        if st.button("➡️ Next: Health Risk Index", type="primary", use_container_width=True):
            st.session_state.current_page = 'health_risk_index'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 健康风险指数页面（第5步）
def show_health_risk_index_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 页面标题
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);">
        <h2 style="color: white; margin: 0;">📈 Step 5: Health Risk Index</h2>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Health risk indices calculated based on previous inputs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 从会话状态获取必要数据
    data = st.session_state.form_data
    
    # 从会话状态获取身高和腰围
    height_cm = data.get('height_cm', 0)
    WC_cm = data.get('WC_cm', 0)
    gender = data.get('gender', 0)
    weight_kg = data.get('weight_kg', 0)
    
    # 计算RFM（根据性别编码：0=男，1=女）
    if height_cm > 0 and WC_cm > 0:
        A = gender
        RFM = 64 - (20 * height_cm / WC_cm) + (12 * A)
        st.session_state.form_data['RFM'] = RFM
    else:
        RFM = 25.0
        st.session_state.form_data['RFM'] = RFM
    
    # 计算TyG指数
    if 'glucose' in data and 'TG' in data and data['TG'] > 0:
        TyG = np.log(data['TG'] * data['glucose'] / 2)
        st.session_state.form_data['TyG'] = TyG
    else:
        TyG = 8.5
        st.session_state.form_data['TyG'] = TyG
    
    # 计算TyG-BMI
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100
        BMI = weight_kg / (height_m ** 2)
        TyG_BMI = TyG * BMI
        st.session_state.form_data['TyG_BMI'] = TyG_BMI
    else:
        TyG_BMI = 212.5
        st.session_state.form_data['TyG_BMI'] = TyG_BMI
    
    # 计算TyG-WHtR
    if WC_cm > 0 and height_cm > 0:
        WHtR = WC_cm / height_cm
        TyG_WHtR = TyG * WHtR
        st.session_state.form_data['TyG_WHtR'] = TyG_WHtR
    else:
        TyG_WHtR = 4.25
        st.session_state.form_data['TyG_WHtR'] = TyG_WHtR
    
    # 显示所有指数
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Relative Fat Mass (RFM)", f"{RFM:.2f}", "Body composition index")
        st.metric("TyG-BMI Index", f"{TyG_BMI:.2f}", "Metabolic and weight index")
    
    with col2:
        st.metric("Triglyceride-glucose Index (TyG)", f"{TyG:.4f}", "Insulin resistance index")
        st.metric("TyG-WHtR Index", f"{TyG_WHtR:.4f}", "Central obesity index")
    
    st.markdown("""
    <div class="info-card slide-in">
        <div style="text-align: center;">
            <h3>✅ All health risk indices have been calculated!</h3>
            <p>Ready for final risk assessment</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Previous: Medical History", use_container_width=True):
            st.session_state.current_page = 'medical_history'
            st.rerun()
    with col3:
        if st.button("🚀 View Risk Assessment Results", type="primary", use_container_width=True):
            st.session_state.current_page = 'results'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 结果页面（第6步）
def show_results_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # 页面标题
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <h2 style="color: white; margin: 0;">📊 Step 6: Risk Assessment Results</h2>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Comprehensive risk evaluation based on all inputs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 执行预测
    perform_prediction()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 执行预测函数
def perform_prediction():
    if model is not None:
        try:
            # 从会话状态获取所有数据
            data = st.session_state.form_data
            
            # 以正确的顺序准备所有需要的特征
            # 连续特征（20个）
            continuous_feature_mapping = {
                'age': float(data.get('age', 70)),
                'temperature': float(data.get('temperature', 36.5)),
                'pulse': float(data.get('pulse', 72)),
                'left_naked_eye': float(data.get('left_naked_eye', 1.0)),
                'right_naked_eye': float(data.get('right_naked_eye', 1.0)),
                'heart_rate': float(data.get('heart_rate', 72)),
                'HGB': float(data.get('HGB', 135.0)),
                'PLT': float(data.get('PLT', 250.0)),
                'Creatinine': float(data.get('Creatinine', 70.0)),
                'Urea': float(data.get('Urea', 5.0)),
                'TC': float(data.get('TC', 4.5)),
                'LDL_C': float(data.get('LDL_C', 2.5)),
                'HDL_C': float(data.get('HDL_C', 1.2)),
                'systolic': float(data.get('systolic', 120)),
                'diastolic': float(data.get('diastolic', 80)),
                'RFM': float(data.get('RFM', 25.0)),
                'TyG': float(data.get('TyG', 8.5)),
                'TyG_BMI': float(data.get('TyG_BMI', 212.5)),
                'TyG_WHtR': float(data.get('TyG_WHtR', 4.25)),
                'count': float(data.get('count', 0))
            }
            
            # 分类特征（8个）- 确保转换为float
            categorical_feature_mapping = {
                'gender': float(data.get('gender', 1)),
                'smoke': float(data.get('smoke', 0)),
                'heart_rhythm': float(data.get('heart_rhythm', 0)),
                'hear': float(data.get('hear', 0)),
                'exercise': float(data.get('exercise', 0)),
                'hypertension_final': float(data.get('hypertension_final', 0)),
                'diabetes_final': float(data.get('diabetes_final', 0)),
                'dyslipidemia_final': float(data.get('dyslipidemia_final', 0))
            }
            
            # 以正确顺序构建输入数组 - 确保所有值都是float类型
            continuous_values = np.array([[
                continuous_feature_mapping['age'],
                continuous_feature_mapping['temperature'],
                continuous_feature_mapping['pulse'],
                continuous_feature_mapping['left_naked_eye'],
                continuous_feature_mapping['right_naked_eye'],
                continuous_feature_mapping['heart_rate'],
                continuous_feature_mapping['HGB'],
                continuous_feature_mapping['PLT'],
                continuous_feature_mapping['Creatinine'],
                continuous_feature_mapping['Urea'],
                continuous_feature_mapping['TC'],
                continuous_feature_mapping['LDL_C'],
                continuous_feature_mapping['HDL_C'],
                continuous_feature_mapping['systolic'],
                continuous_feature_mapping['diastolic'],
                continuous_feature_mapping['RFM'],
                continuous_feature_mapping['TyG'],
                continuous_feature_mapping['TyG_BMI'],
                continuous_feature_mapping['TyG_WHtR'],
                continuous_feature_mapping['count']
            ]], dtype=np.float64)
            
            categorical_values = np.array([[
                categorical_feature_mapping['gender'],
                categorical_feature_mapping['smoke'],
                categorical_feature_mapping['heart_rhythm'],
                categorical_feature_mapping['hear'],
                categorical_feature_mapping['exercise'],
                categorical_feature_mapping['hypertension_final'],
                categorical_feature_mapping['diabetes_final'],
                categorical_feature_mapping['dyslipidemia_final']
            ]], dtype=np.float64)
            
            # === 最小-最大归一化（仅对连续变量）===
            normalized_continuous = (continuous_values - feature_mins) / (feature_maxs - feature_mins)
            normalized_continuous = np.clip(normalized_continuous, 0, 1)

            # 合并归一化的连续变量和原始的分类变量
            input_features_normalized = np.concatenate([normalized_continuous, categorical_values], axis=1)
            
            # 确保输入数据是float类型
            input_features_normalized = input_features_normalized.astype(np.float64)

            # === 使用最优阈值进行预测 ===
            prediction_display = None
            high_risk_prob = 0.5

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_features_normalized)
                high_risk_prob = probabilities[0][1]
                
                # 使用自定义阈值
                best_threshold = custom_threshold  
                
                if high_risk_prob >= best_threshold:
                    prediction_display = 1
                else:
                    prediction_display = 0
                
                st.markdown(f"""
                <div class="info-card slide-in">
                    <div style="text-align: center;">
                        <h4>📊 Using decision threshold: {best_threshold:.3f}</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 如果不支持概率预测，直接使用分类预测
                prediction_display = model.predict(input_features_normalized)[0]
                high_risk_prob = 0.8 if prediction_display == 1 else 0.2
            
            # 显示预测结果
            if prediction_display == 0:
                st.markdown("""
                <div class="success-card slide-in">
                    <div style="text-align: center;">
                        <h1 style="font-size: 2.5em; margin: 10px 0;">✅ Low Risk</h1>
                        <h3>Risk Assessment: Low Risk of Cardiometabolic Multimorbidity</h3>
                        <p>The patient's current profile indicates a low probability of developing cardiometabolic multimorbidity.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-card slide-in pulse">
                    <div style="text-align: center;">
                        <h1 style="font-size: 2.5em; margin: 10px 0;">⚠️ High Risk</h1>
                        <h3>Risk Assessment: High Risk of Cardiometabolic Multimorbidity</h3>
                        <p>The patient's current profile indicates a high probability of developing cardiometabolic multimorbidity.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 显示预测概率
            low_risk_prob = (1 - high_risk_prob) * 100
            high_risk_prob_display = high_risk_prob * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Low Risk Probability", f"{low_risk_prob:.1f}%")
            with col2:
                st.metric("High Risk Probability", f"{high_risk_prob_display:.1f}%")
            
            # 特征重要性显示
            if hasattr(model, 'coef_'):
                st.markdown('<div class="card" style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); margin-top: 30px;"><h3 style="color: white; margin: 0;">🔍 Top 10 Important Features</h3></div>', unsafe_allow_html=True)
                
                # 对于逻辑回归模型，使用系数的绝对值作为重要性
                importances = np.abs(model.coef_[0])
                feature_names = continuous_feature_names + categorical_feature_names
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_[0],
                    'Importance (abs)': importances
                }).sort_values('Importance (abs)', ascending=False).head(10)
                
                st.markdown('<div class="feature-table slide-in">', unsafe_allow_html=True)
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 建议信息
            st.markdown('<div class="card" style="background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); margin-top: 30px;"><h3 style="color: white; margin: 0;">💡 Clinical Recommendations</h3></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card slide-in">
                <ul>
                    <li>This assessment is based on statistical models and should be used as a reference tool</li>
                    <li>Results should be interpreted in conjunction with clinical judgment and other diagnostic information</li>
                    <li>Regular follow-up and monitoring are recommended for all patients</li>
                    <li>Lifestyle modifications and medical interventions should be considered based on individual risk profile</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 添加重新开始评估按钮
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Start New Assessment", type="primary", use_container_width=True, key="restart_button"):
                    st.session_state.current_page = 'demographics'
                    st.session_state.form_data = {}
                    st.rerun()

        except Exception as e:
            st.markdown("""
            <div class="error-card slide-in">
                <h3>❌ Error During Prediction</h3>
                <p>An error occurred during the prediction process. Please try again or contact support.</p>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"Error details: {e}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
    else:
        st.markdown("""
        <div class="error-card slide-in">
            <h3>❌ Model Loading Error</h3>
            <p>The prediction model could not be loaded. Please check the model file and try again.</p>
        </div>
        """, unsafe_allow_html=True)

# 批量预测函数
def show_batch_prediction():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="card" style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);"><h3 style="color: white; margin: 0;">📁 Batch Prediction</h3></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # 检查列名
            expected_columns = all_feature_names
            missing_columns = [col for col in expected_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.markdown(f"""
                <div class="error-card slide-in">
                    <h3>❌ Missing Required Columns</h3>
                    <p>The following columns are missing: {', '.join(missing_columns)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("🚀 Execute Batch Risk Assessment", type="primary", use_container_width=True):
                    with st.spinner('Processing batch prediction...'):
                        # 分离连续变量和分类变量
                        batch_continuous = batch_data[continuous_feature_names].values
                        batch_categorical = batch_data[categorical_feature_names].values
                        
                        # 归一化连续变量
                        batch_normalized = (batch_continuous - feature_mins) / (feature_maxs - feature_mins)
                        batch_normalized = np.clip(batch_normalized, 0, 1)
                        
                        # 合并特征
                        batch_features = np.concatenate([batch_normalized, batch_categorical], axis=1)
                        
                        # 确保数据类型正确
                        batch_features = batch_features.astype(np.float64)
                        
                        # 使用自定义阈值进行预测
                        if hasattr(model, 'predict_proba'):
                            batch_probabilities = model.predict_proba(batch_features)
                            batch_high_risk_probs = batch_probabilities[:, 1]
                            batch_predictions = (batch_high_risk_probs >= custom_threshold).astype(int)
                            
                            batch_data['Predicted Risk Level'] = batch_predictions
                            batch_data['Predicted Risk Level'] = batch_data['Predicted Risk Level'].map({0: 'Low Risk', 1: 'High Risk'})
                            batch_data['Low Risk Probability'] = [f"{prob:.4f}" for prob in batch_probabilities[:, 0]]
                            batch_data['High Risk Probability'] = [f"{prob:.4f}" for prob in batch_probabilities[:, 1]]
                        else:
                            batch_predictions = model.predict(batch_features)
                            batch_data['Predicted Risk Level'] = batch_predictions
                            batch_data['Predicted Risk Level'] = batch_data['Predicted Risk Level'].map({0: 'Low Risk', 1: 'High Risk'})
                        
                        st.markdown('<div class="success-card slide-in">', unsafe_allow_html=True)
                        st.subheader("📊 Batch Prediction Results")
                        st.dataframe(batch_data, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 统计结果
                        st.markdown('<div class="info-card slide-in">', unsafe_allow_html=True)
                        st.write("**Risk Distribution Statistics:**")
                        risk_counts = batch_data['Predicted Risk Level'].value_counts()
                        st.write(risk_counts)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 下载结果
                        import io
                        output = io.BytesIO()
                        batch_data.to_csv(output, index=False, encoding='utf-8-sig')
                        csv_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 Download Prediction Results",
                            data=csv_data,
                            file_name="Cardiometabolic_Multimorbidity_Risk_Assessment_Results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
        except Exception as e:
            st.markdown(f"""
            <div class="error-card slide-in">
                <h3>❌ Error Processing File</h3>
                <p>An error occurred while processing the uploaded file: {e}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 主页面路由
def main():
    # 根据当前页面状态显示相应内容
    if st.session_state.current_page == 'welcome':
        show_welcome_page()
    elif st.session_state.current_page == 'demographics':
        show_demographics_page()
    elif st.session_state.current_page == 'physical_examination':
        show_physical_examination_page()
    elif st.session_state.current_page == 'laboratory_indicators':
        show_laboratory_indicators_page()
    elif st.session_state.current_page == 'medical_history':
        show_medical_history_page()
    elif st.session_state.current_page == 'health_risk_index':
        show_health_risk_index_page()
    elif st.session_state.current_page == 'results':
        show_results_page()
    
    # 仅在欢迎页面显示批量预测
    if st.session_state.current_page == 'welcome':
        show_batch_prediction()

    # 侧边栏信息
    st.sidebar.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.sidebar.header("📖 User Guide")
    
    with st.sidebar.expander("👁️ Vision Conversion Table", expanded=False):
        st.write("""
        **Logarithmic (5-point) vs. Decimal:**
        - 5.3 = 2.0
        - 5.2 = 1.5
        - 5.1 = 1.2
        - 5.0 = 1.0
        - 4.9 = 0.8
        - 4.8 = 0.6
        - 4.7 = 0.5
        - 4.6 = 0.4
        - 4.5 = 0.3
        - 4.4 = 0.25
        - 4.3 = 0.2
        - 4.2 = 0.15
        - 4.1 = 0.12
        - 4.0 = 0.1
        """)
    
    # 侧边栏进度指示器
    if st.session_state.current_page != 'welcome':
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Assessment Progress")
        
        progress_steps = {
            'demographics': ("Demographics", 1, "#4b6cb7"),
            'physical_examination': ("Physical Exam", 2, "#2193b0"),
            'laboratory_indicators': ("Lab Tests", 3, "#cc2b5e"),
            'medical_history': ("Medical History", 4, "#11998e"),
            'health_risk_index': ("Risk Index", 5, "#f46b45"),
            'results': ("Results", 6, "#667eea")
        }
        
        if st.session_state.current_page in progress_steps:
            name, step, color = progress_steps[st.session_state.current_page]
            progress = step / 6
            
            st.sidebar.markdown(f"""
            <div style="margin: 10px 0;">
                <p style="font-size: 14px; margin-bottom: 5px; font-weight: bold;">Step {step}: {name}</p>
                <div style="background-color: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, {color} 0%, {color}99 100%); width: {progress*100}%; height: 100%;"></div>
                </div>
                <p style="text-align: center; font-size: 12px; margin-top: 5px;">{step}/6 ({progress*100:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()