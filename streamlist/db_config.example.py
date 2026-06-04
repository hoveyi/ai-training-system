import pymysql
from sqlalchemy import create_engine
import streamlit as st
import base64
from datetime import datetime
import hashlib

# MySQL数据库配置 — 复制此文件为 db_config.py 并填入你的密码
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'YOUR_PASSWORD_HERE',   # ← 改成你的MySQL密码
    'database': 'users',
    'charset': 'utf8mb4'
}


def get_db_connection():
    """数据库连接函数"""
    return pymysql.connect(**DB_CONFIG)
