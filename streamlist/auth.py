# auth.py - 用户认证模块（MySQL版本）
import streamlit as st
import hashlib
import re
from datetime import datetime
import pandas as pd
from db_config import get_db_connection, init_database
import pymysql
import numpy as np



def hash_password(password):
    """密码哈希加密"""
    return hashlib.sha256(password.encode()).hexdigest()


def validate_email(email):
    """验证邮箱格式"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None


def validate_password_strength(password):
    """验证密码强度"""
    if len(password) < 6:
        return False, "密码长度至少6位"
    if not re.search(r'[A-Za-z]', password):
        return False, "密码必须包含字母"
    if not re.search(r'[0-9]', password):
        return False, "密码必须包含数字"
    return True, "密码强度合格"


def check_login(username, password):
    """验证登录凭证"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")
        hashed_pwd = hash_password(password)

        cursor.execute("""
        SELECT id, username, role, is_active FROM users 
        WHERE username = %s AND password_hash = %s
        """, (username, hashed_pwd))

        user = cursor.fetchone()

        if user and user[3]:  # is_active = True
            # 更新最后登录时间
            cursor.execute("""
            UPDATE users SET last_login = NOW() WHERE id = %s
            """, (user[0],))
            conn.commit()

            # 记录登录活动
            log_activity(user[0], 'login', status='success')

            return True, user[2]  # 返回 role
        else:
            if user:
                log_activity(None, 'login', status='failed', details=f"用户{username}账户未激活")
            return False, None
    finally:
        cursor.close()
        conn.close()


def register_user(username, password, email):
    """注册新用户"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")

        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return False, "用户名已存在"

        # 验证邮箱格式
        if not validate_email(email):
            return False, "邮箱格式不正确"

        # 验证密码强度
        is_valid, msg = validate_password_strength(password)
        if not is_valid:
            return False, msg

        # 创建新用户
        hashed_pwd = hash_password(password)
        cursor.execute("""
        INSERT INTO users (username, password_hash, email, role, created_at) 
        VALUES (%s, %s, %s, 'user', NOW())
        """, (username, hashed_pwd, email))

        conn.commit()

        # 获取新用户ID并记录注册活动
        user_id = cursor.lastrowid
        log_activity(user_id, 'register', status='success')

        return True, "注册成功"
    except Exception as e:
        return False, f"注册失败: {str(e)}"
    finally:
        cursor.close()
        conn.close()


def change_password(username, old_password, new_password):
    """修改密码"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")

        # 验证原密码
        old_hash = hash_password(old_password)
        cursor.execute("""
        SELECT id FROM users WHERE username = %s AND password_hash = %s
        """, (username, old_hash))

        user = cursor.fetchone()
        if not user:
            return False, "原密码错误"

        # 验证新密码强度
        is_valid, msg = validate_password_strength(new_password)
        if not is_valid:
            return False, msg

        # 更新密码
        new_hash = hash_password(new_password)
        cursor.execute("""
        UPDATE users SET password_hash = %s WHERE username = %s
        """, (new_hash, username))

        conn.commit()

        # 记录密码修改活动
        log_activity(user[0], 'change_password', status='success')

        return True, "密码修改成功"
    finally:
        cursor.close()
        conn.close()


def delete_user(username):
    """删除用户（管理员功能）"""
    if username == "admin":
        return False, "不能删除管理员账户"

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")

        # 获取用户ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user:
            # 记录删除活动
            log_activity(user[0], 'delete_user', status='success')

            # 删除用户（级联删除相关活动记录）
            cursor.execute("DELETE FROM users WHERE username = %s", (username,))
            conn.commit()
            return True, "用户删除成功"

        return False, "用户不存在"
    finally:
        cursor.close()
        conn.close()


def get_user_info(username):
    """获取用户信息"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")
        cursor.execute("""
        SELECT id, username, email, role, created_at, last_login, is_active 
        FROM users WHERE username = %s
        """, (username,))

        user = cursor.fetchone()
        return user
    finally:
        cursor.close()
        conn.close()


def get_user_by_id(user_id):
    """根据ID获取用户信息"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")
        cursor.execute("""
        SELECT id, username, email, role, created_at, last_login 
        FROM users WHERE id = %s
        """, (user_id,))

        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()


def log_activity(user_id, activity_type, model_name=None, input_data=None,
                 prediction_result=None, confidence_score=None,
                 processing_time=None, status='success', details=None):
    """记录用户活动"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")
        cursor.execute("""
        INSERT INTO user_activities 
        (user_id, activity_type, model_name, input_data, prediction_result, 
         confidence_score, processing_time, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (user_id, activity_type, model_name, input_data,
              prediction_result, confidence_score, processing_time, status))

        conn.commit()
    except Exception as e:
        print(f"记录活动失败: {e}")
    finally:
        cursor.close()
        conn.close()


def update_model_stats(model_name, confidence_score, processing_time):
    """更新模型使用统计"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")

        # 获取当前统计
        cursor.execute("""
        SELECT total_predictions, avg_confidence, avg_processing_time 
        FROM model_usage_stats WHERE model_name = %s
        """, (model_name,))

        stats = cursor.fetchone()

        if stats:
            total = stats[0] + 1
            new_avg_confidence = (stats[1] * stats[0] + confidence_score) / total
            new_avg_time = (stats[2] * stats[0] + processing_time) / total

            cursor.execute("""
            UPDATE model_usage_stats 
            SET total_predictions = %s, 
                avg_confidence = %s, 
                avg_processing_time = %s,
                last_used = NOW()
            WHERE model_name = %s
            """, (total, new_avg_confidence, new_avg_time, model_name))
        else:
            cursor.execute("""
            INSERT INTO model_usage_stats 
            (model_name, total_predictions, avg_confidence, avg_processing_time, last_used)
            VALUES (%s, 1, %s, %s, NOW())
            """, (model_name, confidence_score, processing_time))

        conn.commit()
    except Exception as e:
        print(f"更新模型统计失败: {e}")
    finally:
        cursor.close()
        conn.close()


def get_user_activities(username, limit=50):
    """获取用户活动记录"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")
        cursor.execute("""
        SELECT ua.*, u.username
        FROM user_activities ua
        JOIN users u ON ua.user_id = u.id
        WHERE u.username = %s
        ORDER BY ua.created_at DESC
        LIMIT %s
        """, (username, limit))

        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_all_users():
    """获取所有用户列表（管理员功能）"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")
        cursor.execute("""
        SELECT id, username, email, role, created_at, last_login, is_active 
        FROM users 
        ORDER BY created_at DESC
        """)

        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_model_stats():
    """获取模型使用统计"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")
        cursor.execute("""
        SELECT * FROM model_usage_stats 
        ORDER BY total_predictions DESC
        """)

        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_system_stats():
    """获取系统整体统计"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")

        stats = {}

        # 总用户数
        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        stats['total_users'] = cursor.fetchone()['total_users']

        # 今日活跃用户
        cursor.execute("""
        SELECT COUNT(DISTINCT user_id) as active_users 
        FROM user_activities 
        WHERE DATE(created_at) = CURDATE()
        """)
        stats['active_users'] = cursor.fetchone()['active_users']

        # 总预测次数
        cursor.execute("""
        SELECT COUNT(*) as total_predictions 
        FROM user_activities 
        WHERE activity_type = 'prediction'
        """)
        stats['total_predictions'] = cursor.fetchone()['total_predictions']

        # 今日预测次数
        cursor.execute("""
        SELECT COUNT(*) as today_predictions 
        FROM user_activities 
        WHERE activity_type = 'prediction' AND DATE(created_at) = CURDATE()
        """)
        stats['today_predictions'] = cursor.fetchone()['today_predictions']

        return stats
    finally:
        cursor.close()
        conn.close()


# 登录UI组件
def login_ui():
    """显示登录界面"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-top: 50px;
    }
    .login-title {
        text-align: center;
        margin-bottom: 2rem;
        color: #4A90E2;
    }
    .login-footer {
        text-align: center;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 登录", "📝 注册"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")

            submitted = st.form_submit_button("登录", width='stretch')

            if submitted:
                if not username or not password:
                    st.error("请填写用户名和密码")
                else:
                    success, role = check_login(username, password)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['user_role'] = role

                        # 获取用户ID
                        user_info = get_user_info(username)
                        if user_info:
                            st.session_state['user_id'] = user_info['id']

                        st.success(f"欢迎回来，{username}！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("用户名", placeholder="3-20个字符", key="reg_user")
            new_email = st.text_input("邮箱", placeholder="your@email.com", key="reg_email")
            new_password = st.text_input("密码", type="password", placeholder="至少6位，包含字母和数字", key="reg_pwd")
            confirm_password = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="reg_confirm")

            submitted_reg = st.form_submit_button("注册", width='stretch')

            if submitted_reg:
                if not new_username or not new_email or not new_password:
                    st.error("请填写所有字段")
                elif new_password != confirm_password:
                    st.error("两次输入的密码不一致")
                else:
                    success, msg = register_user(new_username, new_password, new_email)
                    if success:
                        st.success(msg + "，请登录")
                        st.balloons()
                    else:
                        st.error(msg)


def logout():
    """登出功能"""
    if 'username' in st.session_state:
        user_info = get_user_info(st.session_state['username'])
        if user_info:
            log_activity(user_info['id'], 'logout', status='success')

    for key in ['logged_in', 'username', 'user_role', 'user_id']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def user_profile_ui():
    """用户个人中心UI"""
    if not st.session_state.get('logged_in', False):
        return

    username = st.session_state['username']
    user_info = get_user_info(username)

    if user_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 👤 用户信息")
        st.sidebar.markdown(f"**用户名**: {username}")
        st.sidebar.markdown(f"**角色**: {user_info.get('role', 'user')}")
        st.sidebar.markdown(f"**邮箱**: {user_info.get('email', '未设置')}")
        st.sidebar.markdown(f"**注册时间**: {user_info.get('created_at', '未知')}")

        if user_info.get('last_login'):
            st.sidebar.markdown(f"**最后登录**: {user_info['last_login']}")

        # 修改密码按钮
        if st.sidebar.button("🔑 修改密码", key="change_pwd_btn"):
            st.session_state.show_change_pwd = True

        # 查看使用记录按钮
        if st.sidebar.button("📊 我的使用记录", key="view_activities"):
            st.session_state.show_activities = True

        # 登出按钮
        if st.sidebar.button("🚪 登出", key="logout_btn"):
            logout()

        # 修改密码模态框
        if st.session_state.get('show_change_pwd', False):
            with st.expander("修改密码", expanded=True):
                with st.form("change_password_form"):
                    old_pwd = st.text_input("原密码", type="password")
                    new_pwd = st.text_input("新密码", type="password")
                    confirm_new = st.text_input("确认新密码", type="password")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("确认修改"):
                            if new_pwd != confirm_new:
                                st.error("两次输入的新密码不一致")
                            else:
                                success, msg = change_password(username, old_pwd, new_pwd)
                                if success:
                                    st.success(msg)
                                    st.session_state.show_change_pwd = False
                                    st.rerun()
                                else:
                                    st.error(msg)
                    with col2:
                        if st.form_submit_button("取消"):
                            st.session_state.show_change_pwd = False
                            st.rerun()

        # 查看使用记录
        if st.session_state.get('show_activities', False):
            with st.expander("我的使用记录", expanded=True):
                activities = get_user_activities(username, limit=20)
                if activities:
                    df = pd.DataFrame(activities)
                    df_display = df[['activity_type', 'model_name', 'status', 'created_at']]
                    df_display.columns = ['活动类型', '模型名称', '状态', '时间']
                    st.dataframe(df_display, width='stretch')
                else:
                    st.info("暂无使用记录")

                if st.button("关闭", key="close_activities"):
                    st.session_state.show_activities = False
                    st.rerun()


def require_auth(func):
    """登录验证装饰器"""

    def wrapper(*args, **kwargs):
        if not st.session_state.get('logged_in', False):
            st.warning("请先登录后使用系统功能")
            login_ui()
            return None
        return func(*args, **kwargs)

    return wrapper

def admin_panel():
    """管理员面板"""
    if st.session_state.get('user_role') != 'admin':
        st.error("无权限访问")
        return

    st.markdown("## 👑 管理员面板")

    # 导入图像管理函数
    from image_manager import get_all_classifications, get_classification_detail, delete_classification
    from db_config import get_db_connection
    import pandas as pd
    from datetime import datetime
    import io
    from PIL import Image

    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 用户管理", "📋 使用记录", "📊 模型统计", "📈 系统统计", "📸 图像记录管理"])

    # ==================== 标签页1：用户管理 ====================
    with tab1:
        users = get_all_users()
        if users:
            df = pd.DataFrame(users)
            df_display = df[['id', 'username', 'email', 'role', 'created_at', 'last_login', 'is_active']]
            st.dataframe(df_display, width='stretch')

            # 删除用户功能
            with st.expander("🗑️ 删除用户"):
                user_to_delete = st.selectbox("选择要删除的用户",
                                              [u['username'] for u in users if u['username'] != "admin"])
                if st.button("删除用户", type="secondary"):
                    success, msg = delete_user(user_to_delete)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # ==================== 标签页2：使用记录 ====================
    with tab2:
        st.markdown("### 📋 所有用户活动记录")

        # 筛选器
        col1, col2, col3 = st.columns(3)
        with col1:
            activity_filter = st.selectbox("活动类型",
                                           ["全部", "login", "logout", "prediction", "register", "change_password"])
        with col2:
            model_filter = st.selectbox("模型类型",
                                        ["全部", "flower_classification", "titanic_survival", "fashion_classification",
                                         "nonlinear_regression"])
        with col3:
            status_filter = st.selectbox("状态", ["全部", "success", "failed"])

        # 获取活动记录
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute("USE users")

            query = """
            SELECT ua.*, u.username 
            FROM user_activities ua
            JOIN users u ON ua.user_id = u.id
            WHERE 1=1
            """
            params = []

            if activity_filter != "全部":
                query += " AND ua.activity_type = %s"
                params.append(activity_filter)
            if model_filter != "全部":
                query += " AND ua.model_name = %s"
                params.append(model_filter)
            if status_filter != "全部":
                query += " AND ua.status = %s"
                params.append(status_filter)

            query += " ORDER BY ua.created_at DESC LIMIT 200"

            cursor.execute(query, params)
            activities = cursor.fetchall()

            if activities:
                df = pd.DataFrame(activities)
                df_display = df[
                    ['username', 'activity_type', 'model_name', 'prediction_result', 'confidence_score', 'status',
                     'created_at']]
                df_display.columns = ['用户名', '活动类型', '模型名称', '预测结果', '置信度', '状态', '时间']
                st.dataframe(df_display, width='stretch')

                # 统计信息
                st.markdown("#### 📊 活动统计")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总活动数", len(activities))
                with col2:
                    predictions = len([a for a in activities if a['activity_type'] == 'prediction'])
                    st.metric("预测次数", predictions)
                with col3:
                    success_rate = len([a for a in activities if a['status'] == 'success']) / len(activities) * 100
                    st.metric("成功率", f"{success_rate:.1f}%")
            else:
                st.info("暂无活动记录")
        finally:
            cursor.close()
            conn.close()

    # ==================== 标签页3：模型统计 ====================
    with tab3:
        model_stats = get_model_stats()
        if model_stats:
            df = pd.DataFrame(model_stats)
            st.dataframe(df, width='stretch')

            # 可视化
            st.subheader("📊 模型使用次数对比")
            chart_data = df[['model_name', 'total_predictions']]
            chart_data.columns = ['模型名称', '预测次数']
            st.bar_chart(chart_data.set_index('模型名称'))

            st.subheader("📈 模型平均置信度对比")
            confidence_data = df[['model_name', 'avg_confidence']]
            confidence_data.columns = ['模型名称', '平均置信度']
            st.line_chart(confidence_data.set_index('模型名称'))

    # ==================== 标签页4：系统统计 ====================
    with tab4:
        sys_stats = get_system_stats()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总用户数", sys_stats.get('total_users', 0))
        with col2:
            st.metric("今日活跃用户", sys_stats.get('active_users', 0))
        with col3:
            st.metric("总预测次数", sys_stats.get('total_predictions', 0))
        with col4:
            st.metric("今日预测次数", sys_stats.get('today_predictions', 0))

        # 添加更多统计
        st.markdown("---")
        st.markdown("### 📸 图像分类统计")

        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute("USE users")

            # 图像总数
            cursor.execute("SELECT COUNT(*) as total_images FROM image_classifications")
            total_images = cursor.fetchone()['total_images']

            # 今日上传图像数
            cursor.execute(
                "SELECT COUNT(*) as today_images FROM image_classifications WHERE DATE(created_at) = CURDATE()")
            today_images = cursor.fetchone()['today_images']

            col1, col2 = st.columns(2)
            with col1:
                st.metric("总图像记录数", total_images)
            with col2:
                st.metric("今日上传图像", today_images)

            # 各类别图像数量
            cursor.execute("""
            SELECT predicted_class, COUNT(*) as count 
            FROM image_classifications 
            GROUP BY predicted_class 
            ORDER BY count DESC
            """)
            class_counts = cursor.fetchall()

            if class_counts:
                st.subheader("🌸 各类别图像数量分布")
                df_classes = pd.DataFrame(class_counts)
                st.bar_chart(df_classes.set_index('predicted_class'))
        finally:
            cursor.close()
            conn.close()

    # ==================== 标签页5：图像记录管理（重点功能） ====================
    with tab5:
        st.markdown("### 📸 所有用户图像分类记录")
        st.markdown("查看所有用户上传的图像、预测结果和详细信息")

        # 筛选区域
        st.markdown("#### 🔍 筛选条件")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # 获取所有用户列表
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("USE users")
                cursor.execute("SELECT DISTINCT username FROM image_classifications ORDER BY username")
                users_list = ['全部'] + [row[0] for row in cursor.fetchall()]
            finally:
                cursor.close()
                conn.close()

            selected_user = st.selectbox("👤 选择用户", users_list, key="admin_img_user")

        with col2:
            # 获取所有类别
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("USE users")
                cursor.execute("SELECT DISTINCT predicted_class FROM image_classifications ORDER BY predicted_class")
                classes_list = ['全部'] + [row[0] for row in cursor.fetchall()]
            finally:
                cursor.close()
                conn.close()

            selected_class = st.selectbox("🌸 按类别筛选", classes_list, key="admin_img_class")

        with col3:
            # 置信度阈值
            confidence_threshold = st.slider("🎯 置信度阈值", 0.0, 1.0, 0.0, 0.05)

        with col4:
            sort_by = st.selectbox("📊 排序方式", ["最新优先", "最早优先", "置信度最高", "置信度最低"])

        # 显示数量
        col1, col2 = st.columns([3, 1])
        with col1:
            limit = st.select_slider("显示记录数", options=[20, 50, 100, 200, 500], value=50)
        with col2:
            st.markdown("###")
            refresh_btn = st.button("🔄 刷新", width='stretch')

        # 构建查询
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        try:
            cursor.execute("USE users")

            query = """
            SELECT id, username, predicted_class, confidence_score, 
                   image_filename, image_size, image_type, model_name,
                   processing_time, notes, created_at
            FROM image_classifications 
            WHERE 1=1
            """
            params = []

            if selected_user != '全部':
                query += " AND username = %s"
                params.append(selected_user)

            if selected_class != '全部':
                query += " AND predicted_class = %s"
                params.append(selected_class)

            if confidence_threshold > 0:
                query += " AND confidence_score >= %s"
                params.append(confidence_threshold)

            # 排序
            if sort_by == "最新优先":
                query += " ORDER BY created_at DESC"
            elif sort_by == "最早优先":
                query += " ORDER BY created_at ASC"
            elif sort_by == "置信度最高":
                query += " ORDER BY confidence_score DESC"
            elif sort_by == "置信度最低":
                query += " ORDER BY confidence_score ASC"

            query += f" LIMIT {limit}"

            cursor.execute(query, params)
            records = cursor.fetchall()

            if records:
                st.success(f"✅ 共找到 {len(records)} 条图像记录")

                # 导出功能
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("📥 导出为CSV", key="export_csv"):
                        df_export = pd.DataFrame(records)
                        # 移除二进制数据列
                        if 'image_data' in df_export.columns:
                            df_export = df_export.drop(columns=['image_data'])
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="下载CSV文件",
                            data=csv,
                            file_name=f"classifications_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                # 显示统计摘要
                st.markdown("#### 📊 统计摘要")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    avg_confidence = np.mean([r['confidence_score'] for r in records])
                    st.metric("平均置信度", f"{avg_confidence * 100:.1f}%")
                with summary_col2:
                    unique_users = len(set([r['username'] for r in records]))
                    st.metric("涉及用户数", unique_users)
                with summary_col3:
                    unique_classes = len(set([r['predicted_class'] for r in records]))
                    st.metric("识别类别数", unique_classes)

                st.markdown("---")

                # 分页显示记录
                records_per_page = 5
                total_pages = (len(records) + records_per_page - 1) // records_per_page

                if 'admin_img_page' not in st.session_state:
                    st.session_state.admin_img_page = 0

                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if st.button("◀ 上一页", key="admin_img_prev") and st.session_state.admin_img_page > 0:
                        st.session_state.admin_img_page -= 1
                        st.rerun()

                with col2:
                    st.markdown(f"<center>第 {st.session_state.admin_img_page + 1} / {total_pages} 页</center>",
                                unsafe_allow_html=True)

                with col3:
                    if st.button("下一页 ▶",
                                 key="admin_img_next") and st.session_state.admin_img_page < total_pages - 1:
                        st.session_state.admin_img_page += 1
                        st.rerun()

                # 显示当前页的记录
                start_idx = st.session_state.admin_img_page * records_per_page
                end_idx = min(start_idx + records_per_page, len(records))

                for idx in range(start_idx, end_idx):
                    record = records[idx]

                    # 创建可展开的卡片
                    expander_title = f"📸 {record['predicted_class']} | 置信度: {record['confidence_score'] * 100:.1f}% | 用户: {record['username']} | {record['created_at']}"

                    with st.expander(expander_title, expanded=False):
                        # 获取完整记录（包括图像）
                        detail = get_classification_detail(record['id'], record['username'])

                        if detail:
                            col1, col2 = st.columns([1, 1])

                            with col1:
                                st.markdown("##### 🖼️ 原始图像")
                                if detail.get('image'):
                                    # 调整图像显示大小
                                    st.image(detail['image'], caption=f"上传者: {record['username']}", width=300)
                                else:
                                    st.warning("图像数据不可用")

                            with col2:
                                st.markdown("##### 📋 详细信息")

                                # 创建信息表格
                                info_data = {
                                    "属性": ["记录ID", "用户名", "文件名", "图像类型", "图像大小", "模型名称"],
                                    "值": [
                                        record['id'],
                                        record['username'],
                                        record.get('image_filename', '未知'),
                                        record.get('image_type', '未知'),
                                        f"{record.get('image_size', 0) / 1024:.2f} KB",
                                        record.get('model_name', '未知')
                                    ]
                                }

                                # 预测信息
                                st.markdown("**🎯 预测结果**")
                                pred_col1, pred_col2 = st.columns(2)
                                with pred_col1:
                                    st.markdown(f"- **预测类别**: {record['predicted_class']}")
                                    st.markdown(f"- **置信度**: {record['confidence_score'] * 100:.2f}%")
                                with pred_col2:
                                    st.markdown(f"- **处理时间**: {record.get('processing_time', 0) * 1000:.1f} ms")
                                    st.markdown(f"- **创建时间**: {record['created_at']}")

                                if record.get('notes'):
                                    st.markdown(f"**📝 备注**: {record['notes']}")

                                st.markdown("---")

                                # 显示概率分布
                                if detail.get('probabilities'):
                                    st.markdown("**📊 详细概率分布**")
                                    prob_df = pd.DataFrame({
                                        '类别': list(detail['probabilities'].keys()),
                                        '概率': list(detail['probabilities'].values())
                                    }).sort_values('概率', ascending=False)

                                    # 使用柱状图显示
                                    st.bar_chart(prob_df.set_index('类别'))

                                    # 表格显示
                                    with st.expander("查看详细概率表格"):
                                        st.dataframe(prob_df, width='stretch')

                            # 操作按钮
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button(f"🗑️ 删除此记录", key=f"admin_del_{record['id']}"):
                                    success, msg = delete_classification(record['id'], record['username'])
                                    if success:
                                        st.success(msg)
                                        st.rerun()
                                    else:
                                        st.error(msg)

                            with col2:
                                # 查看用户其他记录
                                if st.button(f"👤 查看 {record['username']} 的其他记录",
                                             key=f"view_user_{record['id']}"):
                                    selected_user = record['username']
                                    st.rerun()

                            with col3:
                                # 查看同类别其他记录
                                if st.button(f"🌸 查看其他 {record['predicted_class']}",
                                             key=f"view_class_{record['id']}"):
                                    selected_class = record['predicted_class']
                                    st.rerun()
            else:
                st.info("📭 暂无符合条件的图像记录")

        except Exception as e:
            st.error(f"查询失败: {str(e)}")
        finally:
            cursor.close()
            conn.close()