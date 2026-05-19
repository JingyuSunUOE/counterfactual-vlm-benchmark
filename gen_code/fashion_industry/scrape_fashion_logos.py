"""
VLM Bias Dataset — Fashion Logo Pipeline
==========================================
Phase 1: SerpAPI + Pexels 爬取真实品牌Logo图 (11品牌 × 5-10张)
Phase 2: GPT + Gemini 双模型 AI 生成 Logo 图
Phase 3: Gemini 3.1 Pro 质量审核

品牌列表 (论文未覆盖):
  Fashion Graphic (3): Ralph Lauren, Burberry, Loewe
  Fashion Monogram (5): Chanel, LV, Gucci GG, YSL, Celine

用法:
    python scrape_fashion_logos.py --phase scrape
    python scrape_fashion_logos.py --phase generate --model both
    python scrape_fashion_logos.py --phase all
    python scrape_fashion_logos.py --brand chanel,lv
    python scrape_fashion_logos.py --count 10 --dry-run
"""

import os, argparse, time
from pathlib import Path

from scraping_utils import (
    load_env, PexelsClient, SerpAPIClient, GeminiReviewer,
    gen_gpt, gen_gemini, download_image, image_hash,
    save_metadata, make_metadata_entry,
)

load_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

REAL_DIR = REPO_ROOT / "dataset" / "fashion_logos_real"
AI_DIR = REPO_ROOT / "dataset" / "fashion_logos_ai"
META_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"

# ═════════════════════════════════════════════════════════════════
# 品牌配置
# ═════════════════════════════════════════════════════════════════

FASHION_GRAPHIC = {
    "ralph_lauren": {
        "name": "Ralph Lauren",
        "search_queries": [
            "Ralph Lauren polo rider logo product",
            "Ralph Lauren polo shirt logo close up",
        ],
        "features": "Polo player on horseback facing RIGHT, swinging a mallet upward. Horse at full gallop.",
        "ai_prompts": [
            "A man wearing a navy polo shirt walking in a park. On his left chest there is a very small embroidered figure of a horseback rider facing RIGHT, swinging a mallet. The logo is tiny, about 2cm. Full body outdoor photography.",
            "Retail display showing folded polo shirts. Each shirt has a tiny embroidered polo rider on the chest facing RIGHT with mallet. The logos are small in the wider shelf scene. Retail store photography.",
            "Close-up of a navy polo shirt chest. Small embroidered silhouette: polo player on horseback at full gallop facing RIGHT, mallet swung upward. ~3cm, dark thread, crisp stitching. Sharp macro focus.",
            "Macro of embroidered polo rider on white cotton. Horseback rider faces RIGHT, arm up with mallet, horse legs stretched running. Navy thread, visible stitches. Shallow depth of field.",
            "A cotton cap with embroidered polo player on front: horseback rider galloping RIGHT, mallet up. Contrasting thread. Close-up merchandise photography.",
        ],
    },
    "burberry": {
        "name": "Burberry",
        "search_queries": [
            "Burberry equestrian knight logo",
            "Burberry trench coat knight emblem close up",
        ],
        "features": "Equestrian knight on horseback facing LEFT, carrying a flag/lance held upward. Horse in walking pose.",
        "ai_prompts": [
            "A person in a trench coat walking on a London street. On a button of the coat there is a tiny embossed emblem of a knight on horseback carrying a flag. The emblem is very small. Street fashion photography.",
            "A luxury store shelf with scarves. One scarf label has a small knight-on-horseback emblem with a flag. Retail interior photography.",
            "Close-up of a trench coat button showing an embossed emblem: an EQUESTRIAN KNIGHT on horseback facing LEFT, carrying a FLAG held upward. Metal relief on a round button. Fashion detail macro.",
            "Close-up of a scarf label with a printed emblem: a knight in armor on horseback facing LEFT, holding a FLAG with a pennant. Fine line art on fabric. Textile detail photography.",
            "Close-up of a leather bag clasp with an engraved knight: armored rider on horseback facing LEFT, FLAG raised. Metal engraving. Luxury accessories macro.",
        ],
    },
    "loewe": {
        "name": "Loewe",
        "search_queries": [
            "Loewe anagram logo four L",
            "Loewe brand logo leather bag",
        ],
        "features": "The Anagram: four capital L letters arranged in a symmetrical cross/diamond pattern, overlapping at the center. Geometric, abstract.",
        "ai_prompts": [
            "A person carrying a tan leather bag on a city street. The bag has a small embossed logo — four L letters arranged in a geometric cross pattern. The logo is tiny in the full-body shot. Street fashion photography.",
            "A leather goods store display. A bag features a small Loewe Anagram — four overlapping L's in a symmetrical diamond shape. Small detail among products. Retail photography.",
            "Close-up of tan leather bag: embossed ANAGRAM logo. FOUR capital L letters arranged symmetrically, overlapping at center in a cross/diamond pattern. Crisp debossing on smooth leather. Product macro.",
            "Close-up of a leather wallet: gold-stamped ANAGRAM. FOUR L letters interlocking in a geometric diamond/cross pattern. Gold on tan leather. Accessories photography.",
            "Macro of fabric label: printed ANAGRAM logo. FOUR L's crossing at center forming a symmetrical geometric pattern. Fine print on white label. Fashion detail.",
        ],
    },
}

FASHION_MONOGRAM = {
    "chanel": {
        "name": "Chanel",
        "search_queries": [
            "Chanel CC interlocking logo bag",
            "Chanel double C logo close up",
        ],
        "features": "Two interlocking C letters BACK-TO-BACK (facing away from each other). Same size, symmetrical. One C opens left, one opens right.",
        "ai_prompts": [
            "A woman carrying a black quilted bag on a city street. The bag's clasp shows two interlocking C's (back-to-back), very small in the full-body street photo. Fashion street photography.",
            "A department store handbag display. A black quilted bag among many others. Its gold CC clasp (two C's back-to-back) is a tiny detail. Store interior photography.",
            "Close-up of gold clasp on black quilted bag: two interlocking C letters BACK-TO-BACK. One C opens LEFT, one opens RIGHT, overlapping in middle. Same size, symmetrical. Polished gold. Luxury macro.",
            "Close-up gold pendant: two C's interlocked BACK-TO-BACK (facing away). One left, one right, overlapping symmetrically. Same size. Gold metal. Jewelry photography.",
            "Close-up compact case lid: two interlocking C's facing AWAY from each other (back-to-back). Same size. Gold on black lacquer. Beauty product photography.",
        ],
    },
    "lv": {
        "name": "Louis Vuitton",
        "search_queries": [
            "Louis Vuitton LV monogram bag",
            "LV logo close up leather",
        ],
        "features": "L and V overlapping, V IN FRONT of L. Classic serif. Flower and star motifs around.",
        "ai_prompts": [
            "A traveler pulling a brown monogram suitcase through an airport. The LV monogram pattern (L and V overlapping, V in front of L) covers the bag. Travel photography.",
            "A luxury store window with multiple bags. A brown monogram bag shows the LV pattern (V over L, flowers, stars). Store photography.",
            "Close-up of brown coated canvas: repeating monogram with L and V overlapping — V IN FRONT of L. Flower and star motifs. Brown/tan. Luxury product photography.",
            "Close-up leather trunk corner: embossed L and V overlap, V IN FRONT of L. Classic serif. Gold on brown leather. Vintage luggage detail.",
            "Close-up wallet surface: printed LV monogram. L and V overlap with V IN FRONT of L. Flower and star motifs around. Brown on tan. Accessories photography.",
        ],
    },
    "gucci_gg": {
        "name": "Gucci GG",
        "search_queries": [
            "Gucci GG interlocking logo bag",
            "Gucci double G monogram close up",
        ],
        "features": "Two interlocking G letters — one upright G and one inverted G, overlapping. The two G's face opposite directions (one up, one down).",
        "ai_prompts": [
            "A person carrying a canvas bag with a repeating GG monogram pattern on a city street. The interlocking G's are small in the full-body shot. Street fashion photography.",
            "A luxury store shelf with belts. One belt buckle shows a gold interlocking GG. The buckle is a small element. Retail photography.",
            "Close-up of a belt buckle: two interlocking G letters. One G is UPRIGHT, the other is INVERTED (upside down), overlapping at center. Gold polished metal. Accessories macro.",
            "Close-up of canvas bag: repeating GG monogram pattern. Each unit shows two G's interlocking — one upright, one inverted. Brown on tan canvas. Product photography.",
            "Close-up of a leather wallet: gold-stamped interlocking GG. Two G's facing opposite directions, overlapping symmetrically. Gold on black leather. Accessories photography.",
        ],
    },
    "ysl": {
        "name": "Yves Saint Laurent (YSL)",
        "search_queries": [
            "YSL logo interlocking letters bag",
            "Saint Laurent YSL monogram close up",
        ],
        "features": "Three letters Y, S, L interlocking vertically. Y on top, S in middle crossing both, L at bottom. All three letters overlap at the center.",
        "ai_prompts": [
            "A woman carrying a black leather bag on a Parisian street. The bag's clasp shows a small gold YSL monogram. The logo is tiny in the full-body shot. Street fashion photography.",
            "A luxury store window with clutch bags. One bag features a gold YSL logo — three interlocking letters. Small detail among products. Retail photography.",
            "Close-up of a black leather bag clasp: gold YSL monogram. Y on TOP, S crossing through the MIDDLE, L at BOTTOM. All three letters INTERLOCKING at center. Polished gold on black. Luxury macro.",
            "Close-up of a pendant: gold YSL. Three letters vertically arranged and overlapping — Y, S, L. Art deco style. Gold metal. Jewelry photography.",
            "Close-up of a wallet: embossed YSL monogram. Y (top), S (middle, crossing), L (bottom). Three letters interlocking. Gold on black grain leather. Accessories photography.",
        ],
    },
    "celine": {
        "name": "Celine",
        "search_queries": [
            "Celine Triomphe bag logo",
            "Celine Triomphe clasp close up",
        ],
        "features": "Triomphe clasp — two symmetrical C letters interlocking face-to-face (NOT back-to-back like Chanel). The C's mirror each other forming a chain-link oval. Often gold hardware on leather.",
        "ai_prompts": [
            "A woman carrying a tan leather shoulder bag on a Parisian street. The bag's clasp shows a small gold Triomphe — two symmetrical C letters interlocking face-to-face. The logo is tiny in the full-body shot. Street fashion photography.",
            "A luxury boutique display with leather bags. One bag features a gold Triomphe clasp — two C letters mirroring each other in a chain-link oval. Small detail among products. Retail photography.",
            "Close-up of a tan leather bag clasp: gold TRIOMPHE hardware. Two symmetrical C letters interlocking FACE-TO-FACE (not back-to-back). The C's mirror each other forming a chain-link OVAL shape. Polished gold on tan leather. Luxury product macro.",
            "Close-up of canvas bag surface: repeating TRIOMPHE monogram pattern. Pairs of C letters facing each other, interlocking in oval chain-link shapes. Brown on tan coated canvas. Product photography.",
            "Macro of gold metal clasp: TRIOMPHE logo. Two C letters facing TOWARD each other, interlocking symmetrically into an oval. Polished gold, fine detail. Accessories photography.",
        ],
    },
}

ALL_BRANDS = {**FASHION_GRAPHIC, **FASHION_MONOGRAM}

def get_brand_category(brand_key):
    if brand_key in FASHION_GRAPHIC:
        return "fashion_graphic"
    if brand_key in FASHION_MONOGRAM:
        return "fashion_monogram"
    return "unknown"

# ═════════════════════════════════════════════════════════════════
# Phase 1: 爬取真实 Logo 图
# ═════════════════════════════════════════════════════════════════
def scrape_fashion_logos(brands=None, count=10, dry_run=False):
    serp = SerpAPIClient()
    pexels = PexelsClient()
    reviewer = GeminiReviewer()
    brand_keys = brands or list(ALL_BRANDS.keys())
    meta = []

    for bk in brand_keys:
        cfg = ALL_BRANDS.get(bk)
        if not cfg:
            print(f"  [SKIP] Unknown brand: {bk}")
            continue

        cat = get_brand_category(bk)
        out_dir = REAL_DIR / bk
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(out_dir.glob(f"{bk}_real_*.jpg"))
        existing_hashes = {image_hash(str(p)) for p in existing}
        start_idx = len(existing) + 1
        needed = max(0, count - len(existing))

        if needed == 0:
            print(f"\n  {cfg['name']}: already have {len(existing)} images, skipping")
            continue

        print(f"\n  {cfg['name']} ({bk}): have {len(existing)}, need {needed} more")

        # SerpAPI 搜索
        candidates = []
        for q in cfg["search_queries"]:
            results = serp.search_images(q, num=20)
            candidates.extend(results)
            if dry_run:
                print(f"    DRY SerpAPI: '{q}' → {len(results)} results")
            time.sleep(1)

        # Pexels fallback
        if len(candidates) < count:
            for q in cfg["search_queries"]:
                pexels_q = q.replace("logo", "").replace("close up", "fashion")
                results = pexels.search(pexels_q, per_page=10)
                for r in results:
                    candidates.append({
                        "url": r["url_large"],
                        "source": "pexels",
                        "title": r.get("photographer", ""),
                        "width": r["width"],
                        "height": r["height"],
                    })
                if dry_run:
                    print(f"    DRY Pexels fallback: '{pexels_q}' → {len(results)} results")

        # 去重 by URL
        seen_urls = set()
        unique = []
        for c in candidates:
            url = c.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(c)

        print(f"    {len(unique)} unique candidates")
        downloaded = 0
        reviewed_pass = 0

        for cand in unique:
            if reviewed_pass >= needed:
                break

            idx = start_idx + downloaded
            fname = f"{bk}_real_{idx:02d}.jpg"
            fpath = out_dir / fname

            if dry_run:
                print(f"    DRY [{idx:02d}]: {cand.get('url', '')[:60]}...")
                downloaded += 1
                reviewed_pass += 1
                continue

            # 下载
            url = cand.get("url", "")
            if not url:
                continue
            print(f"    [{idx:02d}]", end=" ", flush=True)
            ok, w, h = download_image(url, str(fpath), min_size=512)
            if not ok:
                print(f"SKIP (download fail or too small)")
                continue

            # 去重
            h_val = image_hash(str(fpath))
            if h_val in existing_hashes:
                os.remove(str(fpath))
                print("SKIP (duplicate)")
                continue
            existing_hashes.add(h_val)

            # Gemini review: 验证 Logo 可见性
            review = reviewer.review_logo(str(fpath), cfg["name"], cfg["features"])
            passed = review.get("passed", False)
            score = review.get("score", 0)

            if not passed:
                os.remove(str(fpath))
                print(f"REJECT (score={score:.2f}: {review.get('issues', [''])[0][:40]})")
                continue

            print(f"OK ({w}x{h}, review={score:.2f})")
            downloaded += 1
            reviewed_pass += 1
            meta.append(make_metadata_entry(
                filename=fname, category=cat, brand_key=bk,
                source_type="web_scraped", source_url=url,
                license_info=cand.get("license", "fair use for research"),
                width=w, height=h,
                review_score=score,
            ))

        print(f"    Result: {reviewed_pass}/{needed} passed review")

    if meta and not dry_run:
        save_metadata(meta, META_DIR / "fashion_logos_real.json")
    return meta

# ═════════════════════════════════════════════════════════════════
# Phase 2: AI 生成 Logo 图 (双模型)
# ═════════════════════════════════════════════════════════════════
def generate_fashion_logos_ai(brands=None, count=10, model="both", dry_run=False):
    brand_keys = brands or list(ALL_BRANDS.keys())
    reviewer = GeminiReviewer()
    meta_gpt, meta_gemini = [], []

    for bk in brand_keys:
        cfg = ALL_BRANDS.get(bk)
        if not cfg:
            continue

        cat = get_brand_category(bk)
        prompts = cfg["ai_prompts"]

        print(f"\n  {cfg['name']} ({bk}): generating {count} images")

        for model_name in (["gpt", "gemini"] if model == "both" else [model]):
            gen_fn = gen_gpt if model_name == "gpt" else gen_gemini
            out_dir = AI_DIR / bk
            out_dir.mkdir(parents=True, exist_ok=True)
            ok_count, fail_count = 0, 0

            for i in range(count):
                prompt = prompts[i % len(prompts)]
                fname = f"{bk}_ai_{i+1:02d}_{model_name}.png"
                fpath = out_dir / fname

                if fpath.exists():
                    print(f"    [{model_name}] {fname} SKIP")
                    continue
                if dry_run:
                    print(f"    [{model_name}] DRY [{i+1:02d}]: {prompt[:55]}...")
                    continue

                print(f"    [{model_name}] [{i+1:02d}/{count}]", end=" ", flush=True)
                ok, info = gen_fn(prompt, str(fpath))

                review_score = 0
                if ok:
                    review = reviewer.review_logo(str(fpath), cfg["name"], cfg["features"])
                    review_score = review.get("score", 0)
                    print(f"OK ({info}) review={'PASS' if review.get('passed') else 'FAIL'}({review_score:.2f})")
                    ok_count += 1
                else:
                    print(f"FAIL ({info})")
                    fail_count += 1

                entry = make_metadata_entry(
                    filename=fname, category=cat, brand_key=bk,
                    source_type="ai_generated", model=model_name,
                    prompt=prompt, success=ok,
                    review_score=review_score,
                )
                if model_name == "gpt":
                    meta_gpt.append(entry)
                else:
                    meta_gemini.append(entry)
                time.sleep(2)

            print(f"    [{model_name}] OK={ok_count} FAIL={fail_count}")

    if meta_gpt and not dry_run:
        save_metadata(meta_gpt, META_DIR / "fashion_logos_ai_gpt.json")
    if meta_gemini and not dry_run:
        save_metadata(meta_gemini, META_DIR / "fashion_logos_ai_gemini.json")
    return meta_gpt + meta_gemini

# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Fashion Logo Pipeline")
    p.add_argument("--phase", default="all", choices=["scrape", "generate", "all"])
    p.add_argument("--brand", default=None, help="Comma-separated brand keys")
    p.add_argument("--category", default=None, choices=["fashion_graphic", "fashion_monogram"],
                   help="Filter by category")
    p.add_argument("--count", type=int, default=10, help="Images per brand")
    p.add_argument("--model", default="both", choices=["gpt", "gemini", "both"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    brands = None
    if a.brand:
        brands = a.brand.split(",")
    elif a.category:
        if a.category == "fashion_graphic":
            brands = list(FASHION_GRAPHIC.keys())
        else:
            brands = list(FASHION_MONOGRAM.keys())

    if a.phase in ("scrape", "all"):
        print("\n" + "=" * 55 + "\n  Phase 1: Scraping Real Logo Images\n" + "=" * 55)
        scrape_fashion_logos(brands, a.count, a.dry_run)

    if a.phase in ("generate", "all"):
        print("\n" + "=" * 55 + "\n  Phase 2: AI-Generating Logo Images\n" + "=" * 55)
        generate_fashion_logos_ai(brands, a.count, a.model, a.dry_run)

    print("\n  All done!")

if __name__ == "__main__":
    main()
