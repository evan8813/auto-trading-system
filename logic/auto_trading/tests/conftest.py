"""
conftest.py
───────────
pytest 共用設定：將 auto_trading 目錄加入 sys.path，
讓所有測試模組都能直接 import config / models / ... 等。
"""
import sys
from pathlib import Path

# 將 auto_trading 目錄加入路徑（所有模組的進場點）
AUTO_TRADING_DIR = Path(__file__).resolve().parent.parent
if str(AUTO_TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_TRADING_DIR))
