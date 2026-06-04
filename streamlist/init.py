# init_db.py - 独立运行初始化数据库
from db_config import init_database

if __name__ == "__main__":
    print("开始初始化数据库...")
    init_database()
    print("数据库初始化完成！")