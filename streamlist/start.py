import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    # 获取脚本所在目录
    script_path = Path(__file__).resolve()
    return script_path.parent


def check_files():
    """检查必要文件是否存在"""
    root = get_project_root()
    required_files = ['app.py']
    missing_files = []

    for file in required_files:
        if not (root / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"错误: 找不到以下文件: {', '.join(missing_files)}")
        print(f"请确保在项目根目录下运行此脚本")
        return False
    return True


def install_packages():
    """安装依赖包"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'pillow']
    missing_packages = []

    print("[1/3] 检查依赖包...")

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} 已安装")
        except ImportError:
            print(f"  ✗ {package} 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n正在安装缺失的包: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("✓ 所有依赖包安装完成")
        except Exception as e:
            print(f"✗ 安装失败: {e}")
            return False

    return True


def init_database():
    """初始化数据库（可选）"""
    print("\n[2/3] 检查数据库配置...")

    try:
        # 尝试导入数据库配置
        sys.path.insert(0, str(get_project_root()))
        from db_config import init_database
        init_database()
        print("✓ 数据库连接正常")
        return True
    except ImportError:
        print("⚠ 未找到数据库配置，将使用演示模式")
        return False
    except Exception as e:
        print(f"⚠ 数据库初始化失败: {e}")
        print("  将使用演示模式运行")
        return False


def start_streamlit():
    """启动Streamlit应用"""
    print("\n[3/3] 启动Streamlit应用...")
    print()
    print("应用正在启动，浏览器将自动打开...")
    print("如果浏览器未自动打开，请访问: http://localhost:8501")
    print()
    print("按 Ctrl+C 可停止应用")
    print("=" * 50)
    print()

    # 切换到项目目录
    os.chdir(get_project_root())

    # 启动Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n应用已停止")
    except Exception as e:
        print(f"\n启动失败: {e}")
        input("按回车键退出")


def main():
    """主函数"""
    # 设置编码
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print()
    print("=" * 50)
    print("   AI综合应用系统")
    print("   Streamlit 一键启动程序")
    print("=" * 50)
    print()

    # 检查文件
    if not check_files():
        input("按回车键退出")
        return

    # 安装依赖
    if not install_packages():
        input("按回车键退出")
        return

    # 初始化数据库
    init_database()

    # 启动应用
    start_streamlit()


if __name__ == "__main__":
    main()