"""
整理 Fashion 数据集
==================
将 fashion_logos_real/ 和 fashion_logos_ai/ 合并到 fashion_dataset/
统一命名：{brand}_real_01.jpg, {brand}_ai_01.png

用法:
    python organize_fashion.py          # 预览
    python organize_fashion.py --exec   # 执行
"""

import shutil, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

REAL_DIR = REPO_ROOT / "dataset" / "fashion_logos_real"
AI_DIR = REPO_ROOT / "dataset" / "fashion_logos_ai"
OUT_DIR = REPO_ROOT / "dataset" / "fashion_dataset"

BRANDS = ["ralph_lauren", "burberry", "loewe", "chanel", "lv", "gucci_gg", "ysl", "celine"]

def organize(execute=False):
    for brand in BRANDS:
        out = OUT_DIR / brand
        if execute:
            out.mkdir(parents=True, exist_ok=True)

        # --- Real images ---
        real_dir = REAL_DIR / brand
        if real_dir.exists():
            real_files = sorted([f for f in real_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
            print(f"\n=== {brand} === ({len(real_files)} real, ", end="")
            for i, src in enumerate(real_files[:5], 1):
                dst_name = f"{brand}_real_{i:02d}{src.suffix}"
                dst = out / dst_name
                print(f"\n  REAL: {src.name:40s} → {dst_name}", end="")
                if execute:
                    shutil.copy2(str(src), str(dst))

        # --- AI images ---
        ai_dir = AI_DIR / brand
        if ai_dir.exists():
            ai_files = sorted([f for f in ai_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
            print(f"{len(ai_files)} ai)")
            for i, src in enumerate(ai_files[:5], 1):
                dst_name = f"{brand}_ai_{i:02d}.png"
                dst = out / dst_name
                print(f"  AI:   {src.name:40s} → {dst_name}")
                if execute:
                    shutil.copy2(str(src), str(dst))

    # --- Summary ---
    if execute:
        print(f"\n{'='*50}")
        print("验证:")
        for brand in BRANDS:
            d = OUT_DIR / brand
            real_count = len(list(d.glob(f"{brand}_real_*")))
            ai_count = len(list(d.glob(f"{brand}_ai_*")))
            total = real_count + ai_count
            status = "✓" if total == 10 else "✗"
            print(f"  {brand:20s}: {real_count} real + {ai_count} ai = {total} {status}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exec", action="store_true", help="Actually copy files")
    a = p.parse_args()
    organize(a.exec)
    if not a.exec:
        print("\n\n  [预览模式] 加 --exec 执行复制")
