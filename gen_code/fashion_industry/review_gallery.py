"""
生成 HTML 图片审查画廊
=====================
打开生成的 HTML 文件，在浏览器中查看所有原图。

用法:
    python review_gallery.py
    open review_gallery.html
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HTML_PATH = REPO_ROOT / "eval_results" / "fashion_industry" / "reports" / "review_gallery.html"

def collect_images():
    sections = []

    # 行业标准型 - 真实照片
    industry_dir = REPO_ROOT / "dataset" / "industry_originals_clean"
    for cat_dir in sorted(industry_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith('.'):
            continue
        imgs = sorted(cat_dir.glob("*.jpg"))
        if imgs:
            sections.append({
                "title": f"行业标准型 - {cat_dir.name} ({len(imgs)} 张真实照片)",
                "type": "industry_web",
                "images": [{"path": p.resolve().as_uri(), "name": p.name} for p in imgs],
            })

    # Fashion 真实 Logo
    fashion_dir = REPO_ROOT / "dataset" / "fashion_logos_real"
    if fashion_dir.exists():
        for brand_dir in sorted(fashion_dir.iterdir()):
            if not brand_dir.is_dir() or brand_dir.name.startswith('.'):
                continue
            imgs = sorted(brand_dir.glob("*.jpg"))
            if imgs:
                sections.append({
                    "title": f"Fashion Logo - {brand_dir.name} ({len(imgs)} 张真实Logo)",
                    "type": "fashion_real",
                    "images": [{"path": p.resolve().as_uri(), "name": p.name} for p in imgs],
                })

    # AI 生成的原图 (归档中的 vlm_bias_dataset)
    ai_dir = REPO_ROOT / "eval_results" / "fashion_industry" / "archive" / "vlm_bias_dataset" / "original"
    if ai_dir.exists():
        for model_dir in sorted(ai_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for cat_dir in sorted(model_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                imgs = sorted(cat_dir.glob("*.png"))
                if imgs:
                    sections.append({
                        "title": f"AI生成原图 ({model}) - {cat_dir.name} ({len(imgs)} 张)",
                        "type": "ai_original",
                        "images": [{"path": p.resolve().as_uri(), "name": p.name} for p in imgs],
                    })

    # Fashion AI Logo (新生成的)
    ai_fashion_dir = REPO_ROOT / "dataset" / "fashion_logos_ai"
    if ai_fashion_dir.exists():
        for brand_dir in sorted(ai_fashion_dir.iterdir()):
            if not brand_dir.is_dir() or brand_dir.name.startswith('.'):
                continue
            imgs = sorted(brand_dir.glob("*.png"))
            if imgs:
                sections.append({
                    "title": f"Fashion AI Logo - {brand_dir.name} ({len(imgs)} 张)",
                    "type": "fashion_ai",
                    "images": [{"path": p.resolve().as_uri(), "name": p.name} for p in imgs],
                })

    # 已生成的 CF 图 (供参考)
    cf_dir = REPO_ROOT / "cf_dataset" / "industry_cf_web"
    if cf_dir.exists():
        for cat_dir in sorted(cf_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            imgs = sorted(cat_dir.glob("*.png"))
            if imgs:
                sections.append({
                    "title": f"已生成CF (参考) - {cat_dir.name} ({len(imgs)} 张)",
                    "type": "cf_preview",
                    "images": [{"path": p.resolve().as_uri(), "name": p.name} for p in imgs],
                })

    return sections

def generate_html(sections):
    type_colors = {
        "industry_web": "#2196F3",
        "fashion_real": "#9C27B0",
        "ai_original": "#FF9800",
        "fashion_ai": "#E91E63",
        "cf_preview": "#607D8B",
    }

    cards = ""
    nav = ""
    for i, sec in enumerate(sections):
        color = type_colors.get(sec["type"], "#666")
        sid = f"section-{i}"
        nav += f'<a href="#{sid}" style="color:{color};margin:0 8px;font-size:13px;white-space:nowrap">{sec["title"][:40]}</a>\n'

        imgs_html = ""
        for img in sec["images"]:
            imgs_html += f'''
            <div style="display:inline-block;margin:6px;text-align:center;vertical-align:top">
                <img src="{img['path']}" style="max-width:250px;max-height:250px;border:2px solid #ddd;border-radius:4px;cursor:pointer"
                     onclick="this.style.maxWidth=this.style.maxWidth==='250px'?'600px':'250px';this.style.maxHeight=this.style.maxHeight==='250px'?'600px':'250px'">
                <div style="font-size:11px;color:#888;max-width:250px;word-break:break-all">{img['name']}</div>
            </div>'''

        cards += f'''
        <div id="{sid}" style="margin:20px 0;padding:15px;border-left:4px solid {color};background:#fafafa">
            <h3 style="color:{color};margin:0 0 10px">{sec["title"]}</h3>
            <div>{imgs_html}</div>
        </div>'''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLM Bias Dataset - Image Review</title>
<style>body{{font-family:system-ui;max-width:1400px;margin:0 auto;padding:20px}}
img:hover{{border-color:#333!important;box-shadow:0 2px 8px rgba(0,0,0,0.15)}}
</style></head><body>
<h1>VLM Bias Dataset - 图片审查画廊</h1>
<p style="color:#666">点击图片可放大/缩小。审查完毕后告诉我哪些类别/品牌的图片需要替换或补充。</p>
<div style="background:#f0f0f0;padding:10px;border-radius:8px;overflow-x:auto;white-space:nowrap;margin-bottom:20px">
<strong>导航:</strong> {nav}
</div>
<p><strong>统计:</strong> {len(sections)} 个分组, 共 {sum(len(s['images']) for s in sections)} 张图片</p>
{cards}
</body></html>"""
    return html

if __name__ == "__main__":
    sections = collect_images()
    html = generate_html(sections)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html)
    print(f"Generated: {HTML_PATH}")
    print(f"Sections: {len(sections)}, Images: {sum(len(s['images']) for s in sections)}")
