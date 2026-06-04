import streamlit as st
import pymysql
from db_config import get_db_connection
from PIL import Image
import io
import pandas as pd
from datetime import datetime
import json


def save_classified_image(user_id, username, image, filename, predicted_class,
                          confidence_score, class_probabilities, model_name='flower_classification',
                          processing_time=0, notes=''):
    """保存分类后的图像到数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 将图像转换为二进制数据
        img_byte_arr = io.BytesIO()
        if isinstance(image, Image.Image):
            # 调整图像大小以减少存储空间
            image.thumbnail((800, 800))
            image.save(img_byte_arr, format='JPEG', quality=85)
        else:
            image.save(img_byte_arr, format='JPEG', quality=85)

        image_binary = img_byte_arr.getvalue()
        image_size = len(image_binary)

        # 获取图像类型
        if hasattr(image, 'format'):
            image_type = image.format
        else:
            image_type = 'JPEG'

        # 将概率分布转换为JSON字符串
        probs_json = json.dumps(class_probabilities, ensure_ascii=False)

        # 插入数据库
        cursor.execute("USE users")
        cursor.execute("""
        INSERT INTO image_classifications 
        (user_id, username, image_data, image_filename, image_size, image_type, 
         predicted_class, confidence_score, class_probabilities, model_name, 
         processing_time, notes, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (user_id, username, image_binary, filename, image_size, image_type,
              predicted_class, confidence_score, probs_json, model_name,
              processing_time, notes))

        conn.commit()
        return cursor.lastrowid, True, "保存成功"
    except Exception as e:
        print(f"保存图像失败: {e}")
        return None, False, str(e)
    finally:
        cursor.close()
        conn.close()


def get_user_classifications(username, limit=50, offset=0, class_filter=None):
    """获取用户的分类记录"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")

        query = """
        SELECT id, username, predicted_class, confidence_score, 
               image_filename, image_size, image_type, model_name,
               processing_time, notes, created_at
        FROM image_classifications 
        WHERE username = %s
        """
        params = [username]

        if class_filter:
            query += " AND predicted_class = %s"
            params.append(class_filter)

        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_classification_detail(record_id, username):
    """获取分类记录的详细信息（包括图像）"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")
        cursor.execute("""
        SELECT * FROM image_classifications 
        WHERE id = %s AND username = %s
        """, (record_id, username))

        record = cursor.fetchone()

        if record and record['image_data']:
            # 将二进制图像数据转换为PIL Image
            image = Image.open(io.BytesIO(record['image_data']))
            record['image'] = image

            # 解析概率分布
            if record['class_probabilities']:
                record['probabilities'] = json.loads(record['class_probabilities'])

        return record
    finally:
        cursor.close()
        conn.close()


def delete_classification(record_id, username):
    """删除分类记录"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE users")
        cursor.execute("""
        DELETE FROM image_classifications 
        WHERE id = %s AND username = %s
        """, (record_id, username))

        conn.commit()
        return cursor.rowcount > 0, "删除成功" if cursor.rowcount > 0 else "记录不存在"
    except Exception as e:
        return False, str(e)
    finally:
        cursor.close()
        conn.close()


def get_classification_stats(username):
    """获取用户分类统计信息"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")

        # 总分类数
        cursor.execute("""
        SELECT COUNT(*) as total FROM image_classifications WHERE username = %s
        """, (username,))
        total = cursor.fetchone()['total']

        # 各类别统计
        cursor.execute("""
        SELECT predicted_class, COUNT(*) as count, AVG(confidence_score) as avg_confidence
        FROM image_classifications 
        WHERE username = %s
        GROUP BY predicted_class
        ORDER BY count DESC
        """, (username,))

        class_stats = cursor.fetchall()

        # 今日分类数
        cursor.execute("""
        SELECT COUNT(*) as today FROM image_classifications 
        WHERE username = %s AND DATE(created_at) = CURDATE()
        """, (username,))
        today = cursor.fetchone()['today']

        return {
            'total': total,
            'today': today,
            'class_stats': class_stats
        }
    finally:
        cursor.close()
        conn.close()


def get_all_classifications(limit=100, offset=0, class_filter=None):
    """获取所有用户的分类记录（管理员功能）"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        cursor.execute("USE users")

        query = """
        SELECT id, username, predicted_class, confidence_score, 
               image_filename, created_at
        FROM image_classifications 
        WHERE 1=1
        """
        params = []

        if class_filter:
            query += " AND predicted_class = %s"
            params.append(class_filter)

        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def export_user_classifications(username, format='csv'):
    """导出用户分类数据"""
    records = get_user_classifications(username, limit=1000)

    if format == 'csv':
        df = pd.DataFrame(records)
        export_data = df.to_csv(index=False)
        return export_data, 'csv'
    else:
        return records, 'json'