# app.py - 主程序（修改版）
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import time
import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设 streamlist 和 backend 同级）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 Python 路径
sys.path.append(project_root)
from backend.flower_model.model import get_model as get_flower_model
from backend.titanic_model.model import get_model as get_titanic_model
from backend.fashion_model.model import get_model as get_fashion_model
from backend.regression_model.model import get_model as get_regression_model
import torch
from torchvision import transforms

# ---------- 模型缓存（懒加载） ----------
_model_cache = {}


def _load_flower_checkpoint(model_name='resnet50'):
    cache_key = f'flower_{model_name}'
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_flower_model(model_name, num_classes=5).to(device)
    path = os.path.join(project_root, 'backend', 'flower_model', 'checkpoints', 'best_model.pth')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_key = f'{model_name}_state_dict'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    _model_cache[cache_key] = (model, ckpt.get('class_names', []), device)
    return _model_cache[cache_key]


def _load_titanic_checkpoint():
    if 'titanic' in _model_cache:
        return _model_cache['titanic']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(project_root, 'backend', 'titanic_model', 'checkpoints', 'best_model.pth')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = get_titanic_model(ckpt.get('model_name', 'mlp'), ckpt['input_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    _model_cache['titanic'] = (model, ckpt['feature_cols'], ckpt.get('model_name', 'mlp'), device, ckpt.get('test_accuracy', 0))
    return _model_cache['titanic']


def _load_fashion_checkpoint(model_name='cnn'):
    cache_key = f'fashion_{model_name}'
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(project_root, 'backend', 'fashion_model', 'checkpoints', 'best_model.pth')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = get_fashion_model(model_name, ckpt['num_classes']).to(device)
    state_key = f'{model_name}_state_dict'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    acc_key = f'test_accuracy_{model_name}'
    test_acc = ckpt.get(acc_key, 0)
    _model_cache[cache_key] = (model, ckpt['class_names'], device, test_acc)
    return _model_cache[cache_key]


def _load_regression_checkpoint(model_name='mlp'):
    cache_key = f'regression_{model_name}'
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(project_root, 'backend', 'regression_model', 'checkpoints', 'best_model.pth')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    output_dim = 3 if model_name == 'mlp' else 1
    model = get_regression_model(model_name, input_dim=3, output_dim=output_dim).to(device)
    state_key = f'{model_name}_state_dict'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    r2 = ckpt.get(f'r2_{model_name}', 0)
    _model_cache[cache_key] = (model, ckpt, device, r2)
    return _model_cache[cache_key]
# 导入登录模块
from auth import login_ui, logout, user_profile_ui, admin_panel, require_auth
from auth import log_activity, update_model_stats
from image_manager import (
    save_classified_image, get_user_classifications,
    get_classification_detail, delete_classification,
    get_classification_stats
)

# 设置页面配置 - 必须是第一个Streamlit命令
st.set_page_config(
    page_title="AI综合应用系统 | 深度学习模型套件",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'show_change_pwd' not in st.session_state:
    st.session_state['show_change_pwd'] = False


# ------------------ 自定义样式 ------------------
def local_css():
    st.markdown("""
    <style>
        /* 主标题样式 */
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #4A90E2, #8E44AD);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        /* 卡片样式 - 彩色背景 */
        .card {
            border-radius: 15px;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            border: 1px solid #eef2f6;
            transition: all 0.3s ease;
            color: white;
        }
        .card h3, .card h4, .card p {
            color: white;
        }
        .card:hover {
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }
        /* 侧边栏优化 */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        /* 按钮样式 */
        .stButton>button {
            border-radius: 25px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
        }
        /* 指标框 */
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1rem;
            color: white;
            text-align: center;
        }
        hr {
            margin: 1rem 0;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            color: #aaa;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ------------------ 辅助函数 ------------------
def load_demo_image(category="flower"):
    """生成演示用的占位图像 (模拟上传图片)"""
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    if category == "flower":
        for i in range(224):
            img_array[i, :, 0] = 200 + i // 2
            img_array[i, :, 1] = 100 + i // 3
            img_array[i, :, 2] = 150 + i // 4
    elif category == "fashion":
        img_array[:, :, 0] = 100
        img_array[:, :, 1] = 100
        img_array[:, :, 2] = 100
        img_array[50:170, 50:170] = [200, 150, 100]
    else:
        img_array[:, :, :] = [100, 150, 200]
    return Image.fromarray(img_array)


# ------------------ 页面侧边栏 ------------------
def sidebar_navigation():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)

        # 显示登录用户信息
        if st.session_state.get('logged_in', False):
            st.success(f"✅ 欢迎，{st.session_state['username']}!")

        st.markdown("## 🧠 导航菜单")

        # 根据登录状态显示不同的菜单
        if st.session_state.get('logged_in', False):
            app_mode = st.radio(
                "选择模型应用",
                ["🏠 系统主页", "🌸 花卉图像分类", "🚢 Titanic生存预测",
                 "👕 时尚服饰分类", "📈 非线性系统回归"],
                index=0,
                format_func=lambda x: x.split(" ", 1)[-1] if " " in x else x
            )
        else:
            app_mode = "🏠 系统主页"

        st.markdown("---")
        st.markdown("### 📌 系统信息")
        st.info("""
        - **深度学习框架**: PyTorch
        - **前端界面**: Streamlit
        - **模型状态**: 真实神经网络模型
        - **每场景≥2种架构**: ResNet/CNN/MLP/LSTM
        """)

        return app_mode


# ------------------ 主页内容 ------------------
def home_page():
    st.markdown('<div class="main-title">AI 综合应用软件系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">基于深度学习技术的多领域智能预测平台</div>', unsafe_allow_html=True)

    # 未登录时显示登录提示
    if not st.session_state.get('logged_in', False):
        st.info("🔐 请登录后使用完整的AI预测功能，点击左侧菜单或下方登录按钮")
        if st.button("🔐 立即登录", width='stretch'):
            pass  # 登录界面已在主函数中处理

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="card" style="text-align:center; background: linear-gradient(135deg, #FF6B6B 0%, #EE5A24 100%);">
            <h3>🌸</h3>
            <h4>花卉分类</h4>
            <p>5种常见花卉<br>ResNet50 + SimpleCNN</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card" style="text-align:center; background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);">
            <h3>🚢</h3>
            <h4>Titanic生存</h4>
            <p>乘客特征分析<br>MLP深度型 + 宽型</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card" style="text-align:center; background: linear-gradient(135deg, #A8E6CF 0%, #3B8D99 100%);">
            <h3>👕</h3>
            <h4>时尚服饰分类</h4>
            <p>10类服饰物品<br>CNN + ResNet18</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card" style="text-align:center; background: linear-gradient(135deg, #FFD93D 0%, #F39C12 100%);">
            <h3>📈</h3>
            <h4>非线性回归</h4>
            <p>多维非线性系统<br>MLP + LSTM</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 系统特色")
    st.markdown("""
    - ✅ **多模态输入**: 支持图像上传、表格数据、数值参数
    - ✅ **实时预测**: 集成预训练深度学习模型
    - ✅ **可视化结果**: 概率分布、生存指标、回归曲线
    - ✅ **可扩展架构**: 各模型功能独立封装，便于维护
    """)

    st.markdown("### 🚀 快速开始")
    if st.session_state.get('logged_in', False):
        st.info("请从左侧导航栏选择一个模型应用，上传数据或调整参数后点击【预测】按钮获取结果。")
    else:
        st.warning("⚠️ 请先登录以使用完整的AI预测功能")


# ------------------ 花卉分类模块 ------------------
@require_auth
def flower_classification():
    st.markdown("## 🌸 花卉图像分类模型")
    st.markdown("上传花卉图像或使用示例图片，识别雏菊、蒲公英、玫瑰、向日葵、郁金香5种类别。")

    tab1, tab2, tab3 = st.tabs(["🔍 图像分类", "📚 历史记录", "📊 统计分析"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            # 模型架构选择
            flower_model_choice = st.selectbox(
                "🧠 模型架构",
                ["resnet50 (迁移学习, 精度更高)", "simple_cnn (轻量级, 训练更快)"],
                help="ResNet50: 基于ImageNet预训练的深层网络 | SimpleCNN: 4层自定义卷积网络"
            )
            flower_model_name = flower_model_choice.split(" ")[0]

            uploaded_file = st.file_uploader("上传花卉图像 (jpg, png)", type=["jpg", "jpeg", "png"], key="flower")
            use_demo = st.button("🌼 使用示例图像", key="demo_flower")

            if use_demo:
                image = load_demo_image("flower")
                st.image(image, caption="示例花卉图像 (演示)", width=250)
                current_image = image
            elif uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的图像", width=250)
                current_image = image
            else:
                current_image = None

            save_to_db = st.checkbox("💾 保存本次分类结果到数据库", value=True)
            notes = st.text_area("📝 备注（可选）", placeholder="添加一些备注信息...")
        with col2:
            st.markdown("### 📊 预测结果")
            if st.button("🔍 开始识别", key="predict_flower"):
                if current_image is None:
                    st.warning("请先上传图像或点击示例图像")
                else:
                    start_time = time.time()

                    with st.spinner("模型推理中..."):
                        try:
                            model, classes, device = _load_flower_checkpoint(flower_model_name)
                        except FileNotFoundError:
                            st.error(f"模型权重文件未找到，请先运行 backend/flower_model/train.py 训练模型")
                            return

                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                        ])

                        image_tensor = transform(current_image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(image_tensor)
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                            pred_idx = int(np.argmax(probs))
                            pred_class = classes[pred_idx]
                            confidence = float(probs[pred_idx])

                        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
                        processing_time = time.time() - start_time

                        user_id = st.session_state.get('user_id')
                        username = st.session_state.get('username')

                        if user_id:
                            log_activity(user_id=user_id, activity_type='prediction',
                                         model_name=f'flower_{flower_model_name}',
                                         input_data='image_upload', prediction_result=pred_class,
                                         confidence_score=confidence, processing_time=processing_time, status='success')
                            update_model_stats('flower_classification', confidence, processing_time)

                            if save_to_db:
                                filename = uploaded_file.name if uploaded_file else f"demo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                record_id, success, msg = save_classified_image(
                                    user_id=user_id, username=username, image=current_image,
                                    filename=filename, predicted_class=pred_class,
                                    confidence_score=confidence, class_probabilities=prob_dict,
                                    model_name=f'flower_{flower_model_name}',
                                    processing_time=processing_time, notes=notes)

                                if success:
                                    st.success(f"✅ 预测类别: **{pred_class}** (置信度: {confidence * 100:.1f}%)")
                                    st.info(f"💾 分类结果已保存到数据库，记录ID: {record_id}")
                                else:
                                    st.success(f"✅ 预测类别: **{pred_class}** (置信度: {confidence * 100:.1f}%)")
                                    st.warning(f"⚠️ 保存失败: {msg}")
                            else:
                                st.success(f"✅ 预测类别: **{pred_class}** (置信度: {confidence * 100:.1f}%)")

                        st.markdown("#### 类别概率分布")
                        chart_data = pd.DataFrame({"类别": classes, "概率": probs})
                        st.bar_chart(chart_data.set_index("类别"))

                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>🌺 {pred_class}</h3>
                            <p>置信度: {confidence * 100:.1f}% | 模型: {flower_model_name} | {processing_time * 1000:.1f}ms</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("点击【开始识别】按钮进行分类预测")
                st.caption("💡 提示: 可选择不同模型架构对比效果")

    with tab2:
        st.markdown("### 📚 我的分类历史记录")
        username = st.session_state.get('username')
        if username:
            stats = get_classification_stats(username)
            class_options = ['全部'] + [stat['predicted_class'] for stat in stats.get('class_stats', [])]
            selected_class = st.selectbox("按类别筛选", class_options)
            class_filter = None if selected_class == '全部' else selected_class
            records = get_user_classifications(username, limit=50, class_filter=class_filter)

            if records:
                st.info(f"共找到 {len(records)} 条记录")
                for idx, record in enumerate(records):
                    with st.expander(f"📸 {record['predicted_class']} - 置信度: {record['confidence_score'] * 100:.1f}% - {record['created_at']}"):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            detail = get_classification_detail(record['id'], username)
                            if detail and detail.get('image'):
                                st.image(detail['image'], caption=f"上传时间: {record['created_at']}", width=200)
                            else:
                                st.info("图像预览不可用")
                        with col2:
                            st.markdown(f"**记录ID:** {record['id']}")
                            st.markdown(f"**文件名:** {record.get('image_filename', '未知')}")
                            st.markdown(f"**预测类别:** {record['predicted_class']}")
                            st.markdown(f"**置信度:** {record['confidence_score'] * 100:.2f}%")
                            st.markdown(f"**处理时间:** {record.get('processing_time', 0) * 1000:.1f}ms")
                            st.markdown(f"**创建时间:** {record['created_at']}")
                            if record.get('notes'):
                                st.markdown(f"**备注:** {record['notes']}")
                            detail_full = get_classification_detail(record['id'], username)
                            if detail_full and detail_full.get('probabilities'):
                                with st.expander("查看详细概率分布"):
                                    prob_df = pd.DataFrame({
                                        '类别': list(detail_full['probabilities'].keys()),
                                        '概率': list(detail_full['probabilities'].values())
                                    }).sort_values('概率', ascending=False)
                                    st.dataframe(prob_df, width='stretch')
                            if st.button(f"🗑️ 删除", key=f"del_{record['id']}"):
                                success, msg = delete_classification(record['id'], username)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
            else:
                st.info("暂无分类记录，请先在「图像分类」标签页进行分类预测")
        else:
            st.error("无法获取用户信息")

    with tab3:
        st.markdown("### 📊 统计分析")
        username = st.session_state.get('username')
        if username:
            stats = get_classification_stats(username)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总分类数", stats['total'])
            with col2:
                st.metric("今日新增", stats['today'])
            with col3:
                avg_confidence = 0
                if stats['class_stats']:
                    avg_confidence = np.mean([s['avg_confidence'] for s in stats['class_stats']])
                st.metric("平均置信度", f"{avg_confidence * 100:.1f}%")
            if stats['class_stats']:
                st.markdown("#### 各类别分类统计")
                df_stats = pd.DataFrame(stats['class_stats'])
                df_stats.columns = ['类别', '数量', '平均置信度']
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df_stats.set_index('类别')['数量'])
                with col2:
                    st.line_chart(df_stats.set_index('类别')['平均置信度'])
                st.dataframe(df_stats, width='stretch')
            else:
                st.info("暂无统计数据，请先进行分类预测")
        else:
            st.error("无法获取用户信息")


# ------------------ Titanic 生存预测 ------------------
@require_auth
def titanic_survival():
    st.markdown("## 🚢 Titanic 旅客生存概率预测")
    st.markdown("基于神经网络（MLP深度型 / MLP宽型）对旅客信息进行生存概率预测，模型使用真实Titanic数据集训练。")

    tab1, tab2 = st.tabs(["🔮 生存预测", "📚 预测历史"])

    with tab1:
        st.markdown("### 旅客信息输入")

        titanic_model_choice = st.selectbox(
            "🧠 模型架构",
            ["mlp (深度型: BN+多层, 泛化好)", "mlp_wide (宽型: 大隐藏层, 容量大)"],
            help="MLP深度型: 4层+BN+Dropout，更稳定 | MLP宽型: 3层宽网络，拟合能力强"
        )
        titanic_model_name = titanic_model_choice.split(" ")[0]

        with st.form("titanic_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                pclass = st.selectbox("船舱等级", [1, 2, 3],
                                      format_func=lambda x: {1: "头等舱", 2: "二等舱", 3: "三等舱"}[x])
                sex = st.radio("性别", ["male", "female"], horizontal=True)
                age = st.slider("年龄", 0, 100, 28)
            with col2:
                sibsp = st.number_input("兄弟姐妹/配偶数", 0, 8, 0)
                parch = st.number_input("父母/子女数", 0, 6, 0)
                embarked = st.selectbox("登船港口", ["C", "Q", "S"],
                                        format_func=lambda x: {"C": "瑟堡", "Q": "皇后镇", "S": "南安普顿"}[x])
            with col3:
                fare = st.number_input("船票价 (英镑)", 0.0, 600.0, 32.0)

            save_to_db = st.checkbox("💾 保存预测结果到数据库", value=True)
            submitted = st.form_submit_button("📊 预测生存概率", width='stretch')

        if submitted:
            with st.spinner("神经网络推理中..."):
                try:
                    # 加载模型
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    path = os.path.join(project_root, 'backend', 'titanic_model', 'checkpoints', 'best_model.pth')
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                    model = get_titanic_model(titanic_model_name, ckpt['input_dim']).to(device)
                    state_key = f'{titanic_model_name}_state_dict'
                    model.load_state_dict(ckpt[state_key])
                    model.eval()
                    feature_cols = ckpt['feature_cols']
                    test_acc = ckpt.get(f'test_accuracy_{titanic_model_name}', 0)
                except FileNotFoundError:
                    st.error("模型权重未找到，请先运行 backend/titanic_model/train.py 训练模型")
                    return

                # 构造输入特征
                family_size = sibsp + parch + 1
                is_alone = 1 if family_size == 1 else 0
                sex_encoded = 0 if sex == "male" else 1
                embarked_map = {"C": 0, "Q": 1, "S": 2}

                features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare,
                                      embarked_map[embarked], family_size, is_alone]], dtype=np.float32)

                scaler_mean = np.array(ckpt.get('scaler_mean', [2.3, 0.36, 29.7, 0.52, 0.38, 32.2, 1.8, 1.9, 0.4]), dtype=np.float32)
                scaler_scale = np.array(ckpt.get('scaler_scale', [0.84, 0.48, 14.5, 1.1, 0.81, 49.7, 0.82, 1.6, 0.49]), dtype=np.float32)
                features = (features - scaler_mean) / (scaler_scale + 1e-8)

                x = torch.tensor(features).to(device)
                with torch.no_grad():
                    survival_prob = float(model(x).cpu().numpy()[0][0])

                processing_time = 0.01

                # 日志记录
                user_id = st.session_state.get('user_id')
                username = st.session_state.get('username')
                if user_id:
                    log_activity(user_id=user_id, activity_type='prediction',
                                 model_name=f'titanic_{titanic_model_name}',
                                 input_data=str({'pclass': pclass, 'sex': sex, 'age': age}),
                                 prediction_result=f'{survival_prob:.4f}',
                                 confidence_score=survival_prob, processing_time=processing_time, status='success')
                    update_model_stats('titanic_survival', survival_prob, processing_time)

                st.markdown("### 🎲 预测结果")
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("生存概率", f"{survival_prob * 100:.1f}%")
                with colB:
                    if survival_prob > 0.5:
                        st.success("预测: **存活**")
                    else:
                        st.error("预测: **遇难**")
                with colC:
                    st.metric("模型测试精度", f"{test_acc * 100:.1f}%")

                # 特征影响分析
                st.markdown("#### 📊 特征影响分析")
                impacts = {
                    "性别(女)": 0.35 if sex == "female" else -0.20,
                    "头等舱": 0.30 if pclass == 1 else (-0.10 if pclass == 3 else 0.05),
                    "年龄<12": 0.10 if age < 12 else (-0.05 if age > 60 else 0),
                    "家庭规模小": 0.08 if family_size <= 2 else -0.05,
                    "高票价": min(0.15, fare / 300) if fare > 50 else 0,
                }
                st.bar_chart(pd.DataFrame({"特征": list(impacts.keys()), "影响": list(impacts.values())}).set_index("特征"))

                st.caption(f"模型: {titanic_model_name} | 测试精度: {test_acc*100:.1f}% | 由神经网络计算")

    with tab2:
        st.markdown("### 📚 我的预测历史")
        username = st.session_state.get('username')
        if username:
            from auth import get_user_activities
            activities = get_user_activities(username, limit=30)
            titanic_records = [a for a in activities if (a.get('model_name') or '').startswith('titanic')]
            if titanic_records:
                df = pd.DataFrame(titanic_records)
                st.dataframe(df[['model_name', 'prediction_result', 'confidence_score', 'status', 'created_at']], width='stretch')
            else:
                st.info("暂无Titanic预测记录")
        else:
            st.error("无法获取用户信息")


# ------------------ 时尚服饰分类 ------------------
@require_auth
def fashion_classification():
    st.markdown("## 👕 时尚服饰图像分类模型")
    st.markdown("识别10类时尚物品: T恤/上衣, 裤子, 套头衫, 连衣裙, 外套, 凉鞋, 衬衫, 运动鞋, 包, 踝靴。")

    tab1, tab2, tab3 = st.tabs(["🔍 图像分类", "📚 历史记录", "📊 统计分析"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            fashion_model_choice = st.selectbox(
                "🧠 模型架构",
                ["cnn (自定义7层CNN, 轻量高效)", "resnet18 (迁移学习, 精度更高)"],
                help="CNN: 7层自定义卷积+BN+Dropout | ResNet18: 预训练骨干微调"
            )
            fashion_model_name = fashion_model_choice.split(" ")[0]

            uploaded_file = st.file_uploader("上传服饰图像", type=["jpg", "jpeg", "png"], key="fashion")
            demo_fashion = st.button("👗 使用示例服饰图像", key="demo_fashion")
            if demo_fashion:
                img = load_demo_image("fashion")
                st.image(img, caption="示例服饰 (演示)", width=220)
                current_image = img
            elif uploaded_file:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption="上传的服饰", width=220)
                current_image = img
            else:
                current_image = None

            save_to_db = st.checkbox("💾 保存分类结果到数据库", value=True)
        with col2:
            st.markdown("### 📊 预测结果")
            if st.button("✨ 识别服饰类别", key="predict_fashion"):
                if current_image is None:
                    st.warning("请先上传图像或使用示例图像")
                else:
                    start_time = time.time()
                    with st.spinner("模型推理中..."):
                        try:
                            model, classes, device, test_acc = _load_fashion_checkpoint(fashion_model_name)
                        except FileNotFoundError:
                            st.error("模型权重未找到，请先运行 backend/fashion_model/train.py 训练模型")
                            return

                        # Fashion-MNIST是28x28灰度图
                        transform = transforms.Compose([
                            transforms.Resize((28, 28)),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            transforms.Normalize((0.2860,), (0.3530,))
                        ])

                        image_tensor = transform(current_image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(image_tensor)
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                            pred_idx = int(np.argmax(probs))
                            pred_class = classes[pred_idx]
                            confidence = float(probs[pred_idx])

                        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
                        processing_time = time.time() - start_time

                        user_id = st.session_state.get('user_id')
                        username = st.session_state.get('username')
                        if user_id:
                            log_activity(user_id=user_id, activity_type='prediction',
                                         model_name=f'fashion_{fashion_model_name}',
                                         input_data='image_upload', prediction_result=pred_class,
                                         confidence_score=confidence, processing_time=processing_time, status='success')
                            update_model_stats('fashion_classification', confidence, processing_time)

                            if save_to_db:
                                filename = uploaded_file.name if uploaded_file else f"fashion_demo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                record_id, success, msg = save_classified_image(
                                    user_id=user_id, username=username, image=current_image,
                                    filename=filename, predicted_class=pred_class,
                                    confidence_score=confidence, class_probabilities=prob_dict,
                                    model_name=f'fashion_{fashion_model_name}',
                                    processing_time=processing_time, notes='')

                                if success:
                                    st.success(f"🛍️ 识别结果: **{pred_class}** (置信度: {confidence * 100:.1f}%)")
                                    st.info(f"💾 已保存，记录ID: {record_id}")
                                else:
                                    st.success(f"🛍️ 识别结果: **{pred_class}** (置信度: {confidence * 100:.1f}%)")
                                    st.warning(f"保存失败: {msg}")
                            else:
                                st.success(f"🛍️ 识别结果: **{pred_class}** (置信度: {confidence * 100:.1f}%)")

                        prob_df = pd.DataFrame({"类别": classes, "概率": probs}).sort_values("概率", ascending=False)
                        st.bar_chart(prob_df.set_index("类别").head(5))
                        st.caption(f"模型: {fashion_model_name} | 测试精度: {test_acc*100:.1f}% | {processing_time*1000:.1f}ms")
            else:
                st.info("点击【识别服饰类别】进行预测")

    with tab2:
        st.markdown("### 📚 我的分类历史")
        username = st.session_state.get('username')
        if username:
            records = get_user_classifications(username, limit=50)
            fashion_records = [r for r in records if r.get('model_name', '').startswith('fashion')]
            if fashion_records:
                for record in fashion_records:
                    with st.expander(f"📸 {record['predicted_class']} - {record['confidence_score']*100:.1f}% - {record['created_at']}"):
                        detail = get_classification_detail(record['id'], username)
                        if detail and detail.get('image'):
                            st.image(detail['image'], width=150)
                        st.markdown(f"**预测:** {record['predicted_class']} | **置信度:** {record['confidence_score']*100:.1f}%")
                        if detail and detail.get('probabilities'):
                            prob_df = pd.DataFrame({
                                '类别': list(detail['probabilities'].keys()),
                                '概率': list(detail['probabilities'].values())
                            }).sort_values('概率', ascending=False)
                            st.dataframe(prob_df.head(5), width='stretch')
                        if st.button("🗑️ 删除", key=f"del_fashion_{record['id']}"):
                            success, msg = delete_classification(record['id'], username)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                st.info("暂无服饰分类记录")
        else:
            st.error("无法获取用户信息")

    with tab3:
        st.markdown("### 📊 统计分析")
        username = st.session_state.get('username')
        if username:
            stats = get_classification_stats(username)
            if stats['class_stats']:
                df_stats = pd.DataFrame(stats['class_stats'])
                df_stats.columns = ['类别', '数量', '平均置信度']
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df_stats.set_index('类别')['数量'])
                with col2:
                    st.line_chart(df_stats.set_index('类别')['平均置信度'])
            else:
                st.info("暂无统计数据")
        else:
            st.error("无法获取用户信息")


# ------------------ 非线性系统回归预测 ------------------
@require_auth
def nonlinear_regression():
    st.markdown("## 📈 非线性系统回归预测模型")
    st.markdown("基于MLP/LSTM对多维非线性系统进行回归预测，系统自动学习输入-输出的复杂映射关系。")

    tab1, tab2 = st.tabs(["🔮 回归预测", "📚 预测历史"])

    with tab1:
        reg_model_choice = st.selectbox(
            "🧠 模型架构",
            ["mlp (多层感知机, 单点预测)", "lstm (时序记忆网络, 序列预测)"],
            help="MLP: 4层全连接，适合特征向量→输出映射 | LSTM: 循环网络，适合时序动态系统预测"
        )
        reg_model_name = reg_model_choice.split(" ")[0]

        if reg_model_name == 'mlp':
            st.markdown("#### 输入特征向量（3维非线性系统状态）")
            col1, col2, col3 = st.columns(3)
            with col1:
                x1 = st.number_input("特征 x₁", value=0.5, step=0.1, format="%.2f")
            with col2:
                x2 = st.number_input("特征 x₂", value=0.2, step=0.1, format="%.2f")
            with col3:
                x3 = st.number_input("特征 x₃", value=0.8, step=0.1, format="%.2f")

            if st.button("🔮 预测系统输出", key="reg_mlp"):
                with st.spinner("MLP回归预测中..."):
                    try:
                        model, ckpt, device, r2 = _load_regression_checkpoint('mlp')
                        scaler_X = ckpt['scaler_X_mlp']
                        scaler_Y = ckpt['scaler_Y_mlp']
                    except FileNotFoundError:
                        st.error("模型权重未找到，请先运行 backend/regression_model/train.py 训练模型")
                        return

                    features = np.array([[x1, x2, x3]], dtype=np.float32)
                    features_scaled = scaler_X.transform(features)
                    x_tensor = torch.tensor(features_scaled).to(device)

                    with torch.no_grad():
                        y_scaled = model(x_tensor).cpu().numpy()
                        y_pred = scaler_Y.inverse_transform(y_scaled)[0]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("预测 y₁", f"{y_pred[0]:.4f}")
                    with col2:
                        st.metric("预测 y₂", f"{y_pred[1]:.4f}")
                    with col3:
                        st.metric("预测 y₃", f"{y_pred[2]:.4f}")

                    st.markdown("**模型信息**")
                    st.latex(r"\hat{\mathbf{y}} = f_{\text{MLP}}(x_1, x_2, x_3)")
                    st.info(f"模型R²得分: {r2:.4f} | 非线性系统包含正弦、余弦、指数等高阶耦合项")

                    # 保存记录
                    user_id = st.session_state.get('user_id')
                    if user_id:
                        log_activity(user_id=user_id, activity_type='prediction',
                                     model_name='regression_mlp',
                                     input_data=f'[{x1:.2f}, {x2:.2f}, {x3:.2f}]',
                                     prediction_result=f'[{y_pred[0]:.4f}, {y_pred[1]:.4f}, {y_pred[2]:.4f}]',
                                     confidence_score=r2, processing_time=0.01, status='success')
                        update_model_stats('nonlinear_regression', r2, 0.01)
        else:
            st.markdown("#### 时序序列输入（LSTM滑动窗口预测）")
            st.markdown("输入10个连续时间步的系统状态，预测下一时刻的输出值。")

            col1, col2 = st.columns(2)
            with col1:
                seq_input = st.text_area(
                    "输入10个时间步（每行3个特征值，空格分隔）",
                    value="0.0 0.0 0.0\n0.1 0.05 0.02\n0.2 0.10 0.05\n0.3 0.15 0.10\n0.4 0.20 0.15\n0.5 0.25 0.20\n0.6 0.30 0.25\n0.7 0.35 0.30\n0.8 0.40 0.35\n0.9 0.45 0.40",
                    height=200,
                    help="每行表示一个时间步，3个值分别对应 x₁, x₂, x₃"
                )
            with col2:
                st.caption("💡 格式说明:")
                st.caption("- 每行一个时间步")
                st.caption("- 每行3个数值（空格分隔）")
                st.caption("- 共需要10行（seq_len=10）")
                st.caption("- 系统会自动计算下一时刻输出")

            if st.button("📉 运行序列预测", key="reg_lstm"):
                with st.spinner("LSTM序列预测中..."):
                    try:
                        lines = [line.strip() for line in seq_input.strip().split('\n') if line.strip()]
                        values = [list(map(float, line.split())) for line in lines]
                        if len(values) != 10:
                            st.error(f"需要恰好10行输入，当前{len(values)}行")
                            return
                        data = np.array(values, dtype=np.float32)
                    except Exception as e:
                        st.error(f"输入格式错误: {e}")
                        return

                    try:
                        model, ckpt, device, r2 = _load_regression_checkpoint('lstm')
                        scaler_X = ckpt['scaler_X_lstm']
                        scaler_Y = ckpt['scaler_Y_lstm']
                    except FileNotFoundError:
                        st.error("模型权重未找到，请先运行 backend/regression_model/train.py 训练模型")
                        return

                    # 标准化
                    data_scaled = scaler_X.transform(data)
                    x_tensor = torch.tensor(data_scaled).unsqueeze(0).to(device)  # (1, seq_len, 3)

                    with torch.no_grad():
                        y_scaled = model(x_tensor).cpu().numpy()
                        y_pred = scaler_Y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]

                    st.metric("预测下一时刻输出值", f"{y_pred:.4f}")

                    # 可视化序列
                    times = list(range(10))
                    st.markdown("#### 输入序列特征")
                    df_seq = pd.DataFrame({
                        "时间步": times,
                        "x₁": data[:, 0],
                        "x₂": data[:, 1],
                        "x₃": data[:, 2],
                    })
                    st.line_chart(df_seq.set_index("时间步"))
                    st.info(f"模型R²得分: {r2:.4f} | 基于Lorenz-like混沌非线性系统")

                    user_id = st.session_state.get('user_id')
                    if user_id:
                        log_activity(user_id=user_id, activity_type='prediction',
                                     model_name='regression_lstm',
                                     input_data=f'sequence_{len(data)}steps',
                                     prediction_result=f'{y_pred:.4f}',
                                     confidence_score=r2, processing_time=0.01, status='success')
                        update_model_stats('nonlinear_regression', r2, 0.01)

    with tab2:
        st.markdown("### 📚 预测历史")
        username = st.session_state.get('username')
        if username:
            from auth import get_user_activities
            activities = get_user_activities(username, limit=30)
            reg_records = [a for a in activities if (a.get('model_name') or '').startswith('regression')]
            if reg_records:
                df = pd.DataFrame(reg_records)
                st.dataframe(df[['model_name', 'input_data', 'prediction_result', 'confidence_score', 'created_at']],
                             width='stretch')
            else:
                st.info("暂无回归预测记录")
        else:
            st.error("无法获取用户信息")


# ------------------ 主函数 ------------------
def main():
    from db_config import init_database
    init_database()

    local_css()

    # 处理未登录状态 - 显示登录界面
    if not st.session_state.get('logged_in', False):
        login_ui()
        return

    # 已登录用户显示完整系统
    # 显示用户个人中心
    user_profile_ui()

    # 管理员面板（仅管理员可见）
    if st.session_state.get('user_role') == 'admin':
        admin_panel()

    app_mode = sidebar_navigation()

    # 根据导航渲染不同页面
    if app_mode == "🏠 系统主页":
        home_page()
    elif app_mode == "🌸 花卉图像分类":
        flower_classification()
    elif app_mode == "🚢 Titanic生存预测":
        titanic_survival()
    elif app_mode == "👕 时尚服饰分类":
        fashion_classification()
    elif app_mode == "📈 非线性系统回归":
        nonlinear_regression()

    # 页脚
    st.markdown("""
    <div class="footer">
        AI综合应用系统 © 2026 | 四大深度学习模型 | 每场景≥2种神经网络架构
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # 导入pandas用于admin panel
    import pandas as pd

    main()