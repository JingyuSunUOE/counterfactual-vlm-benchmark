"""
整理 Industry 数据集
====================
将 industry_originals_clean/ 和 industry_ai/ 合并到 industry_dataset/
统一命名：{cat}_real_01.jpg, {cat}_ai_01.png
每类取5张real + 5张ai

用法:
    python organize_industry.py          # 预览
    python organize_industry.py --exec   # 执行
"""

import shutil, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

REAL_DIR = REPO_ROOT / "dataset" / "industry_originals_clean"
AI_DIR = REPO_ROOT / "dataset" / "industry_ai"
OUT_DIR = REPO_ROOT / "dataset" / "industry_dataset"

CATEGORIES = [
    "traffic_light", "road_sign", "pingpong_ball", "football",
    "basketball", "tennis_ball", "piano_keyboard", "poker_cards",
]

def organize(execute=False):
    for cat in CATEGORIES:
        out = OUT_DIR / cat
        if execute:
            out.mkdir(parents=True, exist_ok=True)

        # --- Real images ---
        real_dir = REAL_DIR / cat
        real_files = sorted([f for f in real_dir.iterdir()
                             if f.is_file() and not f.name.startswith('.')])[:5]
        print(f"\n=== {cat} === ({len(real_files)} real, ", end="")

        for i, src in enumerate(real_files, 1):
            dst_name = f"{cat}_real_{i:02d}{src.suffix}"
            dst = out / dst_name
            print(f"\n  REAL: {src.name:45s} → {dst_name}", end="")
            if execute:
                shutil.copy2(str(src), str(dst))

        # --- AI images ---
        ai_dir = AI_DIR / cat
        ai_files = sorted([f for f in ai_dir.iterdir()
                           if f.is_file() and not f.name.startswith('.')])[:5]
        print(f" {len(ai_files)} ai)")

        for i, src in enumerate(ai_files, 1):
            dst_name = f"{cat}_ai_{i:02d}.png"
            dst = out / dst_name
            print(f"  AI:   {src.name:45s} → {dst_name}")
            if execute:
                shutil.copy2(str(src), str(dst))

    # --- Summary ---
    if execute:
        print(f"\n{'='*55}")
        print("验证:")
        for cat in CATEGORIES:
            d = OUT_DIR / cat
            real_count = len(list(d.glob(f"{cat}_real_*")))
            ai_count = len(list(d.glob(f"{cat}_ai_*")))
            total = real_count + ai_count
            status = "✓" if total == 10 else "✗"
            print(f"  {cat:20s}: {real_count} real + {ai_count} ai = {total} {status}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exec", action="store_true", help="Actually copy files")
    a = p.parse_args()
    organize(a.exec)
    if not a.exec:
        print("\n\n  [预览模式] 加 --exec 执行复制")
