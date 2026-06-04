import sqlite3
import os
from datetime import datetime


class FlowerDatabase:
    """花卉图片数据库管理"""

    def __init__(self, db_path='flower_classification.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建图片记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flower_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                upload_time TIMESTAMP NOT NULL,
                image_data BLOB
            )
        ''')

        # 创建统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                class_name TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                UNIQUE(date, class_name)
            )
        ''')

        conn.commit()
        conn.close()

    def save_image_record(self, image_path, filename, predicted_class, confidence, image_data=None):
        """保存图片识别记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        upload_time = datetime.now()

        cursor.execute('''
            INSERT INTO flower_images 
            (image_path, filename, predicted_class, confidence, upload_time, image_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_path, filename, predicted_class, confidence, upload_time, image_data))

        # 更新统计
        date_str = upload_time.date().isoformat()
        cursor.execute('''
            INSERT INTO classification_stats (date, class_name, count)
            VALUES (?, ?, 1)
            ON CONFLICT(date, class_name) 
            DO UPDATE SET count = count + 1
        ''', (date_str, predicted_class))

        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        return record_id

    def get_all_records(self, limit=100):
        """获取所有识别记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, image_path, filename, predicted_class, confidence, upload_time
            FROM flower_images
            ORDER BY upload_time DESC
            LIMIT ?
        ''', (limit,))

        records = cursor.fetchall()
        conn.close()

        return records

    def get_record_by_id(self, record_id):
        """根据ID获取记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, image_path, filename, predicted_class, confidence, upload_time, image_data
            FROM flower_images
            WHERE id = ?
        ''', (record_id,))

        record = cursor.fetchone()
        conn.close()

        return record

    def get_statistics(self):
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT class_name, COUNT(*) as total
            FROM flower_images
            GROUP BY class_name
            ORDER BY total DESC
        ''')

        stats = cursor.fetchall()
        conn.close()

        return stats

    def search_by_class(self, class_name):
        """根据花卉类别搜索图片"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, image_path, filename, predicted_class, confidence, upload_time
            FROM flower_images
            WHERE predicted_class = ?
            ORDER BY upload_time DESC
        ''', (class_name,))

        records = cursor.fetchall()
        conn.close()

        return records

    def delete_record(self, record_id):
        """删除记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 先获取图片路径
        cursor.execute('SELECT image_path FROM flower_images WHERE id = ?', (record_id,))
        result = cursor.fetchone()

        if result:
            image_path = result[0]
            # 删除数据库记录
            cursor.execute('DELETE FROM flower_images WHERE id = ?', (record_id,))
            conn.commit()
            conn.close()

            # 删除图片文件
            if os.path.exists(image_path):
                os.remove(image_path)

            return True

        conn.close()
        return False


# 创建全局数据库实例
db = FlowerDatabase()