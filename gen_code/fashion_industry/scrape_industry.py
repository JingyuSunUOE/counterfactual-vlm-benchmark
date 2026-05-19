"""
VLM Bias Dataset — 行业标准型 Pipeline
======================================
Phase 1: Pexels 爬取真实照片 (8类 × 5-10张)
Phase 2: 双模型 (GPT + Gemini) 生成反事实图
Phase 3: Gemini 3.1 Pro 质量审核

用法:
    python scrape_industry.py --phase scrape
    python scrape_industry.py --phase generate --model both
    python scrape_industry.py --phase review
    python scrape_industry.py --phase all
    python scrape_industry.py --category traffic_light --count 5
    python scrape_industry.py --dry-run
"""

import os, argparse, time
from pathlib import Path
from datetime import datetime

from scraping_utils import (
    load_env, PexelsClient, GeminiImageEditor, GeminiReviewer,
    gen_gpt, gen_gemini, download_image, image_hash,
    save_metadata, make_metadata_entry,
)

load_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

ORIG_DIR = REPO_ROOT / "dataset" / "industry_originals_clean"
CF_DIR = REPO_ROOT / "cf_dataset" / "industry_cf_web"
AI_DIR = REPO_ROOT / "dataset" / "industry_ai"
META_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"

# ── 搜索配置 ──────────────────────────────────────────────────────
INDUSTRY_SEARCH = {
    "traffic_light": {
        "queries": ["traffic light red yellow green", "traffic signal intersection close up"],
    },
    "road_sign": {
        "queries": ["red stop sign road intersection", "octagonal stop sign close up"],
    },
    "pingpong_ball": {
        "queries": ["table tennis ball white", "ping pong ball close up"],
    },
    "football": {
        "queries": ["soccer ball black white pentagon", "classic football close up grass"],
    },
    "basketball": {
        "queries": ["orange basketball court", "basketball close up texture orange"],
    },
    "tennis_ball": {
        "queries": ["tennis ball yellow green felt", "tennis ball close up court"],
    },
    "piano_keyboard": {
        "queries": ["piano keyboard black white keys", "piano keys close up"],
    },
    "poker_cards": {
        "queries": ["playing cards hearts spades red black", "poker cards suits close up"],
    },
}

# ── 反事实编辑 Prompt (image-to-image) ────────────────────────────
# 改写自 gen_cf_final.py 的 CF_INDUSTRY，适配真实照片编辑
EDIT_PROMPTS = {
    "traffic_light": {
        "mods": [
            {"id": 1, "type": "order_swap", "desc": "绿上黄中红下",
             "gt": {"order": "green-yellow-red"}, "bias": {"order": "red-yellow-green"},
             "prompt": "Edit this traffic light image: rearrange the lights so GREEN is on TOP, YELLOW in the MIDDLE, and RED on the BOTTOM. Keep everything else in the scene exactly the same."},
            {"id": 2, "type": "order_swap", "desc": "黄上红中绿下",
             "gt": {"order": "yellow-red-green"}, "bias": {"order": "red-yellow-green"},
             "prompt": "Edit this traffic light image: rearrange the lights so YELLOW is on TOP, RED in the MIDDLE, and GREEN on the BOTTOM. Keep everything else unchanged."},
            {"id": 3, "type": "order_swap", "desc": "红上绿中黄下",
             "gt": {"order": "red-green-yellow"}, "bias": {"order": "red-yellow-green"},
             "prompt": "Edit this traffic light image: swap the YELLOW and GREEN positions so the order is RED on TOP, GREEN in the MIDDLE, YELLOW on the BOTTOM. Keep everything else unchanged."},
        ],
    },
    "road_sign": {
        "mods": [
            {"id": 1, "type": "color_change", "desc": "STOP牌绿底",
             "gt": {"color": "green"}, "bias": {"color": "red"},
             "prompt": "Edit this STOP sign: change the background color from RED to GREEN. Keep the white 'STOP' text and octagonal shape. Keep everything else unchanged."},
            {"id": 2, "type": "color_change", "desc": "STOP牌蓝底",
             "gt": {"color": "blue"}, "bias": {"color": "red"},
             "prompt": "Edit this STOP sign: change the background color from RED to BLUE. Keep the white 'STOP' text and octagonal shape. Keep everything else unchanged."},
            {"id": 3, "type": "color_change", "desc": "STOP牌黄底",
             "gt": {"color": "yellow"}, "bias": {"color": "red"},
             "prompt": "Edit this STOP sign: change the background color from RED to YELLOW, and change the text color to BLACK. Keep the octagonal shape. Keep everything else unchanged."},
        ],
    },
    "pingpong_ball": {
        "mods": [
            {"id": 1, "type": "color_change", "desc": "绿色乒乓球",
             "gt": {"color": "green"}, "bias": {"color": "white/orange"},
             "prompt": "Edit this image: change the ping pong ball color to bright GREEN. Keep everything else the same."},
            {"id": 2, "type": "color_change", "desc": "蓝色乒乓球",
             "gt": {"color": "blue"}, "bias": {"color": "white/orange"},
             "prompt": "Edit this image: change the ping pong ball color to bright BLUE. Keep everything else the same."},
            {"id": 3, "type": "color_change", "desc": "紫色乒乓球",
             "gt": {"color": "purple"}, "bias": {"color": "white/orange"},
             "prompt": "Edit this image: change the ping pong ball color to bright PURPLE. Keep everything else the same."},
        ],
    },
    "football": {
        "mods": [
            {"id": 1, "type": "color_change", "desc": "全红足球",
             "gt": {"color": "all red"}, "bias": {"color": "black and white"},
             "prompt": "Edit this soccer ball: change ALL panels to RED. No black pentagons or white hexagons — the entire ball should be solid RED. Keep everything else the same."},
            {"id": 2, "type": "color_change", "desc": "全蓝足球",
             "gt": {"color": "all blue"}, "bias": {"color": "black and white"},
             "prompt": "Edit this soccer ball: change ALL panels to BLUE. No black or white panels — the entire ball should be solid BLUE. Keep everything else the same."},
        ],
    },
    "basketball": {
        "mods": [
            {"id": 1, "type": "color_change", "desc": "蓝色篮球",
             "gt": {"color": "blue"}, "bias": {"color": "orange"},
             "prompt": "Edit this basketball: change the color from ORANGE to BLUE. Keep the black seam lines and pebbled texture. Keep everything else the same."},
            {"id": 2, "type": "color_change", "desc": "绿色篮球",
             "gt": {"color": "green"}, "bias": {"color": "orange"},
             "prompt": "Edit this basketball: change the color from ORANGE to GREEN. Keep the black seam lines and pebbled texture. Keep everything else the same."},
        ],
    },
    "tennis_ball": {
        "mods": [
            {"id": 1, "type": "color_change", "desc": "红色网球",
             "gt": {"color": "red"}, "bias": {"color": "yellow-green"},
             "prompt": "Edit this tennis ball: change the color from YELLOW-GREEN to bright RED. Keep the fuzzy felt texture and white seam. Keep everything else the same."},
            {"id": 2, "type": "color_change", "desc": "蓝色网球",
             "gt": {"color": "blue"}, "bias": {"color": "yellow-green"},
             "prompt": "Edit this tennis ball: change the color from YELLOW-GREEN to bright BLUE. Keep the fuzzy felt texture and white seam. Keep everything else the same."},
        ],
    },
    "piano_keyboard": {
        "mods": [
            {"id": 1, "type": "color_invert", "desc": "黑白反转",
             "gt": {"color": "inverted"}, "bias": {"color": "standard"},
             "prompt": "Edit this piano keyboard: INVERT the key colors. Make the natural keys BLACK and the sharp/flat keys WHITE. Keep everything else the same."},
            {"id": 2, "type": "pattern_change", "desc": "黑键4个一组",
             "gt": {"grouping": "4"}, "bias": {"grouping": "2 and 3"},
             "prompt": "Edit this piano keyboard: change the black key grouping so every group has FOUR black keys together (not the standard 2-and-3 pattern). Keep everything else the same."},
        ],
    },
    "poker_cards": {
        "mods": [
            {"id": 1, "type": "color_swap", "desc": "黑心红桃",
             "gt": {"heart": "black", "spade": "red"}, "bias": {"heart": "red", "spade": "black"},
             "prompt": "Edit these playing cards: swap the colors so HEARTS are BLACK and SPADES are RED. Reverse the standard color assignment. Keep everything else the same."},
            {"id": 2, "type": "color_swap", "desc": "黑方红梅",
             "gt": {"diamond": "black", "club": "red"}, "bias": {"diamond": "red", "club": "black"},
             "prompt": "Edit these playing cards: swap the colors so DIAMONDS are BLACK and CLUBS are RED. Reverse the standard color assignment. Keep everything else the same."},
        ],
    },
}

# ═════════════════════════════════════════════════════════════════
# Phase 1: 爬取
# ═════════════════════════════════════════════════════════════════
def scrape_industry(categories=None, count=10, dry_run=False):
    pexels = PexelsClient()
    meta = []
    cats = categories or list(INDUSTRY_SEARCH.keys())

    for cat in cats:
        cfg = INDUSTRY_SEARCH.get(cat)
        if not cfg:
            print(f"  [SKIP] Unknown category: {cat}")
            continue
        out_dir = ORIG_DIR / cat
        out_dir.mkdir(parents=True, exist_ok=True)

        # 已有图片
        existing = sorted(out_dir.glob(f"{cat}_web_*.jpg"))
        existing_hashes = {image_hash(str(p)) for p in existing}
        # 基于最大编号计算下一个索引，避免覆盖
        max_idx = 0
        for p in existing:
            try:
                idx_str = p.stem.split("_web_")[1]
                max_idx = max(max_idx, int(idx_str))
            except (IndexError, ValueError):
                pass
        start_idx = max(len(existing) + 1, max_idx + 1)
        needed = max(0, count - len(existing))

        if needed == 0:
            print(f"\n  {cat}: already have {len(existing)} images, skipping")
            continue

        print(f"\n  {cat}: have {len(existing)}, need {needed} more")

        # 搜索
        candidates = []
        for q in cfg["queries"]:
            results = pexels.search(q, per_page=15)
            candidates.extend(results)
            if dry_run:
                print(f"    DRY: '{q}' → {len(results)} results")

        # 去重 by pexels id
        seen_ids = set()
        unique = []
        for c in candidates:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                unique.append(c)

        print(f"    {len(unique)} unique candidates")
        downloaded = 0

        for photo in unique:
            if downloaded >= needed:
                break
            idx = start_idx + downloaded
            fname = f"{cat}_web_{idx:02d}.jpg"
            fpath = out_dir / fname

            if dry_run:
                print(f"    DRY [{idx:02d}]: {photo['url'][:70]}...")
                downloaded += 1
                continue

            print(f"    [{idx:02d}]", end=" ", flush=True)
            ok, w, h = pexels.download(photo, str(fpath), min_size=512)
            if not ok:
                print(f"SKIP (too small: {w}x{h})")
                continue

            # 去重 hash
            h_val = image_hash(str(fpath))
            if h_val in existing_hashes:
                os.remove(str(fpath))
                print("SKIP (duplicate)")
                continue
            existing_hashes.add(h_val)

            print(f"OK ({w}x{h})")
            downloaded += 1
            meta.append(make_metadata_entry(
                filename=fname, category="industry", brand_key=cat,
                source_type="web_scraped", source_url=photo["pexels_url"],
                license_info=photo["license"],
                photographer=photo["photographer"],
                width=w, height=h,
            ))

        print(f"    Downloaded: {downloaded}/{needed}")

    if meta and not dry_run:
        save_metadata(meta, META_DIR / "industry_web_scrape.json")
    return meta

# ── AI 原图生成 Prompt (text-to-image) ───────────────────────────
# 复用 gen_originals_final.py 的 INDUSTRY prompt，生成标准原图
INDUSTRY_AI_PROMPTS = {
    "traffic_light": [
        "A wide street photograph of a busy city intersection. A traffic light is visible in the upper portion of the frame, relatively small against the urban backdrop. The signal shows three lights vertically: RED on top, YELLOW in middle, GREEN on bottom. Cars and pedestrians are in the scene. Daytime street photography.",
        "A suburban road scene with a traffic light in the distance, mounted on a pole. The signal clearly shows RED (top), YELLOW (center), GREEN (bottom). Trees and houses visible. The traffic light is a small element in the wider landscape. Daytime photography.",
        "A close-up photograph of a standard vertical traffic light. Three circular lights stacked: RED on top, YELLOW/AMBER middle, GREEN bottom. All three colors vivid and distinct. Metal housing, clear sky behind. Sharp focus.",
        "Close-up of a modern LED traffic signal: RED (top), YELLOW (middle), GREEN (bottom). LED dots visible inside each lens. Clean housing. Isolated against sky.",
        "A traffic light at eye level showing three lights vertically: RED top, YELLOW center, GREEN bottom. The red light is currently illuminated. Clear daytime visibility.",
    ],
    "road_sign": [
        "A wide shot of a residential intersection. A red octagonal STOP sign is visible on a pole at the corner, relatively small in the frame. Houses, trees, parked cars fill the scene. The sign reads 'STOP' in white on RED background. Street photography.",
        "A highway scene with multiple road signs visible in the distance. A RED octagonal STOP sign and a blue direction sign mounted on poles. The signs are small elements in the wider road landscape. Driving photography.",
        "A close-up of an octagonal STOP sign. Bright RED background with 'STOP' in large WHITE capitals. White border. Clean, undamaged. Clear sky. Sharp focus photograph.",
        "Close-up of a speed limit sign: white circle with RED border, number '30' in BLACK. Clean and legible. School zone setting. Road sign photography.",
        "A standard triangular yield sign with RED border and WHITE center, shot at close range. Clear detail. Mounted on pole at junction. Sign photography.",
    ],
    "pingpong_ball": [
        "A wide shot of a table tennis match in a sports hall. Two players at opposite ends of the table. A small WHITE ball is visible mid-flight above the green table. The ball is tiny in the frame. Sports event photography.",
        "A table tennis table in a recreation room. A WHITE ping pong ball sits on the green surface near the net. The ball is small relative to the full table view. Wide-angle indoor sports photography.",
        "Close-up of a single WHITE table tennis ball on a green surface. Perfectly spherical, smooth, matte white, ~40mm. Sharp macro focus. Sports photography.",
        "An ORANGE table tennis ball held between two fingers. Standard competition bright ORANGE, smooth and round. Clean studio background. Product photography.",
        "Close-up of three WHITE ping pong balls arranged on a green table near the net. Standard 40mm white. Well-lit facility. Sports photography.",
    ],
    "football": [
        "A wide shot of a football match on a grass pitch. Players running, a BLACK AND WHITE soccer ball visible on the grass, small in the frame. The ball shows the classic black pentagon / white hexagon pattern. Stadium crowd in background. Sports event photography.",
        "A park scene with children playing football in the distance. A traditional BLACK AND WHITE soccer ball on the grass. The ball is a small element in the wider park landscape. Outdoor recreational photography.",
        "Close-up of a classic soccer ball on green grass. WHITE hexagonal panels and BLACK pentagonal panels in the traditional pattern. Crisp black and white. Sharp focus sports photography.",
        "A person's foot about to kick a BLACK AND WHITE soccer ball. Classic pattern clearly visible. Action sports photography on grass.",
        "Close-up of a soccer ball in a goal net. Classic BLACK AND WHITE pattern: black pentagons, white hexagons. Stadium background. Professional photography.",
    ],
    "basketball": [
        "A wide shot of a basketball game in a gymnasium. Players on court, an ORANGE basketball mid-air, small in the frame. The ball is standard ORANGE with BLACK seam lines. Indoor arena sports photography.",
        "An outdoor basketball court in a park. An ORANGE basketball sitting on the concrete near the hoop. The ball is small in the wide scene. Urban outdoor sports photography.",
        "Close-up of a standard ORANGE basketball on a hardwood court. BLACK seam lines creating the 8-panel pattern. Pebbled texture. Sharp focus sports photography.",
        "Close-up texture of a basketball surface. ORANGE pebbled leather with a BLACK channel line. Warm saturated orange, deep black groove. Studio macro photography.",
        "An ORANGE basketball held in one hand. Bright ORANGE with BLACK channel lines. Pebbled texture visible. Clean gym background. Sports photography.",
    ],
    "tennis_ball": [
        "A wide shot of a tennis match on a clay court. A player serving, a small FLUORESCENT YELLOW-GREEN ball visible in the air. The ball is tiny in the frame against the court and crowd. Sports event photography.",
        "A tennis court overview. Two YELLOW-GREEN tennis balls on the baseline, small in the wide frame. Red clay surface, net visible. Sports facility photography.",
        "Close-up of a FLUORESCENT YELLOW-GREEN tennis ball on a hard court. Fuzzy felt surface, white curved seam line. Vivid optic yellow-green. Sharp sports photography.",
        "A tennis ball held before serving. FLUORESCENT YELLOW-GREEN fuzzy felt, ~6.5cm. Bright optic yellow-green clearly visible. Close-up action sports photo.",
        "Two YELLOW-GREEN tennis balls next to a racket on grass. Regulation fluorescent color, fuzzy felt. Close-up on green grass. Wimbledon-style photography.",
    ],
    "piano_keyboard": [
        "A wide shot of a concert hall with a grand piano on stage. The keyboard is visible in the distance showing WHITE natural keys and BLACK sharp keys in groups of 2 and 3. The piano is a medium element in the grand hall. Concert photography.",
        "A living room scene with an upright piano against the wall. The keyboard shows WHITE keys in front, BLACK keys raised behind in 2-and-3 grouping. The piano is part of the room setting. Interior photography.",
        "Close-up of a piano keyboard: wide WHITE natural keys, narrower BLACK sharp/flat keys raised behind in groups of TWO and THREE. Multiple octaves. Sharp focus music photography.",
        "Top-down close-up of piano keys. WHITE keys longer and wider. BLACK keys in repeating groups: 2 black, then 3 black. Grand piano polished surface. Music photography.",
        "A pianist's hands on standard keyboard. WHITE keys lower, BLACK keys raised in 2-then-3 grouping. Clean polished keys. Performance photography.",
    ],
    "poker_cards": [
        "A wide shot of a poker table with multiple players. Playing cards scattered on the green felt, some showing RED hearts/diamonds, others BLACK spades/clubs. The cards are small in the casino scene. Casino photography.",
        "A card game in progress on a dining table. Several cards face-up showing RED heart symbols and BLACK spade symbols. Cards are medium-small in the domestic scene. Casual game photography.",
        "Close-up of four cards fanned out: heart (RED), diamond (RED), spade (BLACK), club (BLACK). Hearts/diamonds=RED, spades/clubs=BLACK. White background. Card photography.",
        "Two aces side by side: Ace of Hearts with RED heart, Ace of Spades with BLACK spade. Clear RED and BLACK distinction. On green felt. Casino close-up.",
        "Four Aces in a row: Hearts (RED), Diamonds (RED), Spades (BLACK), Clubs (BLACK). Red ink and black ink distinct. Product card photography.",
    ],
}

# ═════════════════════════════════════════════════════════════════
# Phase 1.5: AI 原图生成（非反事实）
# ═════════════════════════════════════════════════════════════════
def generate_industry_ai(categories=None, count=5, model="both", dry_run=False):
    cats = categories or list(INDUSTRY_AI_PROMPTS.keys())
    meta_gpt, meta_gemini = [], []

    for cat in cats:
        prompts = INDUSTRY_AI_PROMPTS.get(cat)
        if not prompts:
            continue

        print(f"\n  {cat}: generating {count} AI images")

        for model_name in (["gpt", "gemini"] if model == "both" else [model]):
            gen_fn = gen_gpt if model_name == "gpt" else gen_gemini
            out_dir = AI_DIR / cat
            out_dir.mkdir(parents=True, exist_ok=True)
            ok_count, fail_count = 0, 0

            for i in range(count):
                prompt = prompts[i % len(prompts)]
                fname = f"{cat}_ai_{i+1:02d}_{model_name}.png"
                fpath = out_dir / fname

                if fpath.exists():
                    print(f"    [{model_name}] {fname} SKIP")
                    continue
                if dry_run:
                    print(f"    [{model_name}] DRY [{i+1:02d}]: {prompt[:55]}...")
                    continue

                print(f"    [{model_name}] [{i+1:02d}/{count}]", end=" ", flush=True)
                ok, info = gen_fn(prompt, str(fpath))

                if ok:
                    print(f"OK ({info})")
                    ok_count += 1
                else:
                    print(f"FAIL ({info})")
                    fail_count += 1

                entry = make_metadata_entry(
                    filename=fname, category="industry", brand_key=cat,
                    source_type="ai_generated", model=model_name,
                    prompt=prompt, success=ok,
                )
                if model_name == "gpt":
                    meta_gpt.append(entry)
                else:
                    meta_gemini.append(entry)
                time.sleep(2)

            print(f"    [{model_name}] OK={ok_count} FAIL={fail_count}")

    if meta_gpt and not dry_run:
        save_metadata(meta_gpt, META_DIR / "industry_ai_gpt.json")
    if meta_gemini and not dry_run:
        save_metadata(meta_gemini, META_DIR / "industry_ai_gemini.json")
    return meta_gpt + meta_gemini


# ═════════════════════════════════════════════════════════════════
# Phase 2: 双模型生成反事实图
# ═════════════════════════════════════════════════════════════════
def generate_industry_cf(categories=None, model="both", dry_run=False):
    cats = categories or list(EDIT_PROMPTS.keys())
    editor = GeminiImageEditor()
    meta_gpt, meta_gemini = [], []

    for cat in cats:
        cfg = EDIT_PROMPTS.get(cat)
        if not cfg:
            continue

        # 找所有 web 原图
        orig_dir = ORIG_DIR / cat
        originals = sorted(orig_dir.glob(f"{cat}_web_*.jpg"))
        if not originals:
            print(f"\n  {cat}: no original images found, skipping")
            continue

        print(f"\n{'='*50}\n  {cat} ({len(originals)} originals × {len(cfg['mods'])} mods)\n{'='*50}")

        for orig_path in originals:
            orig_name = orig_path.stem  # e.g. traffic_light_web_01
            for mod in cfg["mods"]:
                # Gemini edit
                if model in ("both", "gemini"):
                    fname_g = f"{orig_name}_cf_mod{mod['id']}_{mod['type']}_gemini.png"
                    fpath_g = CF_DIR / cat / fname_g
                    fpath_g.parent.mkdir(parents=True, exist_ok=True)

                    if fpath_g.exists():
                        print(f"    [Gemini] {fname_g} SKIP")
                    elif dry_run:
                        print(f"    [Gemini] DRY: {orig_name} mod{mod['id']} {mod['desc']}")
                    else:
                        print(f"    [Gemini] {orig_name} mod{mod['id']} {mod['desc']}", end=" ", flush=True)
                        ok, info = editor.edit(str(orig_path), mod["prompt"], str(fpath_g))
                        print(f"{'OK' if ok else 'FAIL'} ({info})")
                        meta_gemini.append(make_metadata_entry(
                            filename=fname_g, category="industry", brand_key=cat,
                            source_type="cf_from_web", model="gemini",
                            prompt=mod["prompt"], success=ok,
                            mod_id=mod["id"], mod_type=mod["type"],
                            mod_desc=mod["desc"],
                            ground_truth=mod["gt"], bias_answer=mod["bias"],
                            original_image=orig_path.name,
                        ))
                        time.sleep(2)

                # GPT (text-to-image with scene description, since GPT edit API is limited)
                if model in ("both", "gpt"):
                    fname_gpt = f"{orig_name}_cf_mod{mod['id']}_{mod['type']}_gpt.png"
                    fpath_gpt = CF_DIR / cat / fname_gpt
                    fpath_gpt.parent.mkdir(parents=True, exist_ok=True)

                    # GPT text-to-image: 构造描述场景 + 修改的完整 prompt
                    gpt_prompt = f"A realistic photograph of a {cat.replace('_', ' ')}. {mod['prompt'].replace('Edit this', 'Show a').replace('Keep everything else unchanged.', 'Photorealistic, high quality.')} "

                    if fpath_gpt.exists():
                        print(f"    [GPT]    {fname_gpt} SKIP")
                    elif dry_run:
                        print(f"    [GPT]    DRY: {orig_name} mod{mod['id']} {mod['desc']}")
                    else:
                        print(f"    [GPT]    {orig_name} mod{mod['id']} {mod['desc']}", end=" ", flush=True)
                        ok, info = gen_gpt(gpt_prompt, str(fpath_gpt))
                        print(f"{'OK' if ok else 'FAIL'} ({info})")
                        meta_gpt.append(make_metadata_entry(
                            filename=fname_gpt, category="industry", brand_key=cat,
                            source_type="cf_from_web", model="gpt",
                            prompt=gpt_prompt, success=ok,
                            mod_id=mod["id"], mod_type=mod["type"],
                            mod_desc=mod["desc"],
                            ground_truth=mod["gt"], bias_answer=mod["bias"],
                            original_image=orig_path.name,
                        ))
                        time.sleep(2)

    if meta_gemini and not dry_run:
        save_metadata(meta_gemini, META_DIR / "industry_web_cf_gemini.json")
    if meta_gpt and not dry_run:
        save_metadata(meta_gpt, META_DIR / "industry_web_cf_gpt.json")
    return meta_gemini + meta_gpt

# ═════════════════════════════════════════════════════════════════
# Phase 3: Gemini Review
# ═════════════════════════════════════════════════════════════════
def review_industry_cf(categories=None, retry_failed=True):
    reviewer = GeminiReviewer()
    editor = GeminiImageEditor()
    cats = categories or list(EDIT_PROMPTS.keys())
    results = []

    for cat in cats:
        cfg = EDIT_PROMPTS.get(cat)
        if not cfg:
            continue
        cf_cat_dir = CF_DIR / cat
        if not cf_cat_dir.exists():
            continue

        print(f"\n  Reviewing {cat}...")
        cf_images = sorted(cf_cat_dir.glob("*.png"))

        for cf_path in cf_images:
            # 从文件名解析原图和 mod
            parts = cf_path.stem.split("_cf_mod")
            if len(parts) != 2:
                continue
            orig_stem = parts[0]  # e.g. traffic_light_web_01
            orig_path = ORIG_DIR / cat / f"{orig_stem}.jpg"
            if not orig_path.exists():
                print(f"    [SKIP] original not found: {orig_path}")
                continue

            # 解析 mod_id
            mod_info = parts[1]  # e.g. 1_order_swap_gemini
            mod_id = int(mod_info.split("_")[0])
            mod = next((m for m in cfg["mods"] if m["id"] == mod_id), None)
            if not mod:
                continue

            print(f"    {cf_path.name}", end=" ", flush=True)
            review = reviewer.review_counterfactual(
                str(cf_path), str(orig_path), mod["desc"], mod["gt"]
            )
            passed = review.get("passed", False)
            score = review.get("score", 0)
            print(f"{'PASS' if passed else 'FAIL'} (score={score:.2f})")

            # Retry if failed
            if not passed and retry_failed:
                stronger_prompt = mod["prompt"] + " Make sure the modification is clearly visible and obvious. The change must be unmistakable."
                retry_path = cf_path.with_name(cf_path.stem + "_retry.png")
                print(f"      Retrying...", end=" ", flush=True)
                ok, _ = editor.edit(str(orig_path), stronger_prompt, str(retry_path))
                if ok:
                    review2 = reviewer.review_counterfactual(
                        str(retry_path), str(orig_path), mod["desc"], mod["gt"]
                    )
                    if review2.get("passed", False):
                        # 替换原文件
                        os.replace(str(retry_path), str(cf_path))
                        review = review2
                        print(f"PASS on retry (score={review2.get('score', 0):.2f})")
                    else:
                        os.remove(str(retry_path))
                        review["needs_manual_review"] = True
                        print(f"STILL FAIL")
                else:
                    print("retry generation failed")
                    review["needs_manual_review"] = True

            results.append({
                "file": cf_path.name,
                "category": cat,
                "mod_id": mod_id,
                "mod_desc": mod["desc"],
                "original": orig_path.name,
                **review,
                "timestamp": datetime.now().isoformat(),
            })

    if results:
        save_metadata(results, META_DIR / "review_results.json")
    return results

# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Industry Standard Pipeline")
    p.add_argument("--phase", default="all", choices=["scrape", "generate_ai", "generate", "review", "all"])
    p.add_argument("--category", default=None, help="Comma-separated categories")
    p.add_argument("--count", type=int, default=10, help="Images per category (scrape phase)")
    p.add_argument("--model", default="both", choices=["gpt", "gemini", "both"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cats = a.category.split(",") if a.category else None

    if a.phase in ("scrape", "all"):
        print("\n" + "=" * 55 + "\n  Phase 1: Scraping from Pexels\n" + "=" * 55)
        scrape_industry(cats, a.count, a.dry_run)

    if a.phase in ("generate_ai", "all"):
        print("\n" + "=" * 55 + "\n  Phase 1.5: Generating AI Originals\n" + "=" * 55)
        generate_industry_ai(cats, a.count, a.model, a.dry_run)

    if a.phase in ("generate", "all"):
        print("\n" + "=" * 55 + "\n  Phase 2: Generating Counterfactuals\n" + "=" * 55)
        generate_industry_cf(cats, a.model, a.dry_run)

    if a.phase in ("review", "all"):
        if not a.dry_run:
            print("\n" + "=" * 55 + "\n  Phase 3: Gemini 3.1 Pro Review\n" + "=" * 55)
            review_industry_cf(cats)
        else:
            print("\n  [DRY] Skipping review phase")

    print("\n  All done!")

if __name__ == "__main__":
    main()
