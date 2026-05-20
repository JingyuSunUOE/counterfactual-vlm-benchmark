"""
Industry CF 反事实图生成
========================
基于 industry_dataset/ 中的 10张原图(5 real + 5 ai) × mods × 2 models

Gemini: image-to-image 编辑（保留原图背景）
GPT: text-to-image 生成（CF prompt）

用法:
    python gen_industry_cf.py --model both
    python gen_industry_cf.py --model gemini --category traffic_light,road_sign
    python gen_industry_cf.py --dry-run
"""

import os, json, time, base64, argparse, requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DATASET_DIR = REPO_ROOT / "dataset" / "industry_dataset"
CF_DIR = REPO_ROOT / "cf_dataset" / "industry_cf"
META_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"

load_dotenv(REPO_ROOT / ".env", override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

# ═══════════════════════════════════════════════════════════════
# CF 定义：每类别 2-3 mods
# ═══════════════════════════════════════════════════════════════

INDUSTRY_CF = {
    "traffic_light": {"name": "交通灯", "mods": [
        {"id": 1, "type": "order_swap", "desc": "绿上黄中红下",
         "gt": {"order": "green-yellow-red"}, "bias": {"order": "red-yellow-green"},
         "edit_prompt": "Edit this traffic light image: rearrange the lights so GREEN is on TOP, YELLOW in MIDDLE, RED on BOTTOM. Reverse the standard order. Keep everything else unchanged.",
         "prompts": [
             "A busy intersection with a traffic light. The signal shows GREEN on TOP, YELLOW in MIDDLE, RED on BOTTOM. Green at the top. Cars and buildings around. Daytime street photography.",
             "Close-up of a traffic light: GREEN on TOP, YELLOW MIDDLE, RED BOTTOM. Top light is green, middle amber, bottom red. Modern LED. Clear sky behind.",
             "A traffic light over a suburban road: GREEN on TOP, then YELLOW, then RED at bottom. Green in the highest position. Daytime.",
             "Close-up traffic signal mounted on pole: GREEN (top), YELLOW (center), RED (bottom). Non-standard arrangement. Clear background.",
         ]},
        {"id": 2, "type": "order_swap", "desc": "黄上红中绿下",
         "gt": {"order": "yellow-red-green"}, "bias": {"order": "red-yellow-green"},
         "edit_prompt": "Edit this traffic light image: rearrange so YELLOW is on TOP, RED in MIDDLE, GREEN on BOTTOM. Keep everything else unchanged.",
         "prompts": [
             "A street intersection with traffic light showing YELLOW on TOP, RED MIDDLE, GREEN BOTTOM. Amber at the top. Urban background.",
             "Close-up: YELLOW on TOP, RED MIDDLE, GREEN BOTTOM. Top light is amber. Modern signal. Clear background.",
             "Traffic light over road: YELLOW, RED, GREEN from top to bottom. Yellow highest. Daytime.",
         ]},
        {"id": 3, "type": "order_swap", "desc": "红上绿中黄下",
         "gt": {"order": "red-green-yellow"}, "bias": {"order": "red-yellow-green"},
         "edit_prompt": "Edit this traffic light image: swap the middle and bottom lights. GREEN should be in MIDDLE, YELLOW on BOTTOM. Red stays on top. Keep everything else unchanged.",
         "prompts": [
             "Close-up traffic light: RED TOP, GREEN MIDDLE, YELLOW BOTTOM. Middle is green not yellow, bottom is yellow not green.",
             "Traffic signal at intersection: RED top, GREEN center, YELLOW bottom. Green and yellow reversed from normal. Urban scene.",
         ]},
    ]},
    "road_sign": {"name": "STOP路牌", "mods": [
        {"id": 1, "type": "color_change", "desc": "STOP牌绿底",
         "gt": {"color": "green"}, "bias": {"color": "red"},
         "edit_prompt": "Edit this image: change the STOP sign background color from RED to GREEN. Keep the white text 'STOP' and octagonal shape. Everything else unchanged.",
         "prompts": [
             "A residential intersection. An octagonal STOP sign on a pole, but the background color is GREEN instead of red. 'STOP' in white on GREEN. Houses and trees around.",
             "Close-up octagonal STOP sign: 'STOP' in WHITE on GREEN background (NOT red). Standard shape, GREEN color. Pole-mounted, clear sky.",
             "Close-up: GREEN octagonal stop sign. White 'STOP' text, white border. Green background instead of red.",
             "A GREEN STOP sign at a junction. Octagonal, white text on green. Not the normal red. Roadside setting.",
         ]},
        {"id": 2, "type": "color_change", "desc": "STOP牌蓝底",
         "gt": {"color": "blue"}, "bias": {"color": "red"},
         "edit_prompt": "Edit this image: change the STOP sign background color from RED to BLUE. Keep the white text 'STOP' and octagonal shape. Everything else unchanged.",
         "prompts": [
             "An intersection with a BLUE octagonal STOP sign. 'STOP' in white on BLUE. Street scene with buildings.",
             "Close-up: octagonal 'STOP' sign, BLUE background, white letters. Standard shape but BLUE not red.",
             "BLUE octagonal stop sign at road junction. White 'STOP' on BLUE. Residential setting.",
         ]},
        {"id": 3, "type": "color_change", "desc": "STOP牌黄底",
         "gt": {"color": "yellow"}, "bias": {"color": "red"},
         "edit_prompt": "Edit this image: change the STOP sign background color from RED to YELLOW. Change the text to BLACK. Keep the octagonal shape. Everything else unchanged.",
         "prompts": [
             "Close-up: octagonal STOP sign, YELLOW background with BLACK text 'STOP'. Standard shape, YELLOW color.",
             "YELLOW octagonal stop sign. Black 'STOP' on YELLOW. Unusual color. Roadside.",
         ]},
    ]},
    "basketball": {"name": "篮球", "mods": [
        {"id": 1, "type": "color_change", "desc": "蓝色篮球",
         "gt": {"color": "blue"}, "bias": {"color": "orange"},
         "edit_prompt": "Edit this image: change the basketball color from orange to BLUE. The ball should be bright BLUE with black seam lines. Keep everything else unchanged.",
         "prompts": [
             "Basketball game in gym. A BLUE basketball (not orange) mid-air. Blue with black seams. Sports event photography.",
             "Close-up: BLUE basketball on hardwood. Bright BLUE not orange, black seam lines. Pebbled texture.",
             "Player holding BLUE basketball. BLUE with black grooves. Not orange. Gym background.",
             "Close-up BLUE basketball surface. Pebbled, black channel lines. BLUE not orange. Macro.",
         ]},
        {"id": 2, "type": "color_change", "desc": "绿色篮球",
         "gt": {"color": "green"}, "bias": {"color": "orange"},
         "edit_prompt": "Edit this image: change the basketball color from orange to GREEN. The ball should be bright GREEN with black seam lines. Keep everything else unchanged.",
         "prompts": [
             "Close-up: GREEN basketball on court. Bright GREEN not orange, black channel lines. Pebbled.",
             "Player dribbling GREEN basketball. GREEN with black grooves. Not orange. Gym.",
             "GREEN basketball texture close-up. Pebbled, black seams. GREEN not orange.",
         ]},
        {"id": 3, "type": "color_change", "desc": "紫色篮球",
         "gt": {"color": "purple"}, "bias": {"color": "orange"},
         "edit_prompt": "Edit this image: change the basketball color from orange to PURPLE. The ball should be bright PURPLE with black seam lines. Keep everything else unchanged.",
         "prompts": [
             "Close-up: PURPLE basketball on court. Bright PURPLE not orange, black channel lines. Pebbled texture.",
             "Player holding PURPLE basketball. PURPLE with black grooves. Not orange. Gym background.",
             "PURPLE basketball on hardwood. Vivid PURPLE, black seams. Not orange. Macro.",
         ]},
    ]},
    "tennis_ball": {"name": "网球", "mods": [
        {"id": 1, "type": "color_change", "desc": "红色网球",
         "gt": {"color": "red"}, "bias": {"color": "yellow-green"},
         "edit_prompt": "Edit this image: change the tennis ball color to bright RED. The ball should be RED fuzzy felt with white seam. Not yellow-green. Keep everything else unchanged.",
         "prompts": [
             "Tennis match on court. A RED ball (not yellow-green) visible in play. Sports event photography.",
             "Close-up: RED tennis ball on court. Bright RED fuzzy felt, white seam. Not yellow-green.",
             "Player holding RED tennis ball. RED felt, not fluorescent yellow. Close-up.",
             "Two RED tennis balls next to racket on grass. Bright RED felt.",
         ]},
        {"id": 2, "type": "color_change", "desc": "蓝色网球",
         "gt": {"color": "blue"}, "bias": {"color": "yellow-green"},
         "edit_prompt": "Edit this image: change the tennis ball color to bright BLUE. The ball should be BLUE fuzzy felt. Not yellow-green. Keep everything else unchanged.",
         "prompts": [
             "Close-up: BLUE tennis ball on court. Bright BLUE fuzzy felt. Not yellow-green.",
             "Player tossing BLUE tennis ball. BLUE felt. Not yellow. Sports.",
             "BLUE tennis ball on grass. Bright BLUE fuzzy felt. Not yellow-green.",
         ]},
        {"id": 3, "type": "color_change", "desc": "粉色网球",
         "gt": {"color": "pink"}, "bias": {"color": "yellow-green"},
         "edit_prompt": "Edit this image: change the tennis ball color to bright PINK. The ball should be PINK fuzzy felt. Not yellow-green. Keep everything else unchanged.",
         "prompts": [
             "Close-up: PINK tennis ball on court. Bright PINK fuzzy felt, white seam. Not yellow-green.",
             "Player holding PINK tennis ball. PINK felt. Not yellow. Close-up.",
             "PINK tennis ball on grass. Bright PINK fuzzy felt. Not yellow-green.",
         ]},
    ]},
    "piano_keyboard": {"name": "钢琴键盘", "mods": [
        {"id": 1, "type": "color_invert", "desc": "黑白反转",
         "gt": {"color": "inverted"}, "bias": {"color": "standard"},
         "edit_prompt": "Edit this image: INVERT the piano keyboard colors. Natural keys should be BLACK, sharp/flat keys should be WHITE. Reverse the standard black-and-white pattern. Keep everything else unchanged.",
         "prompts": [
             "A concert hall with a piano on stage. The keyboard has INVERTED colors — BLACK natural keys, WHITE sharp keys. The reversed keyboard is part of the stage scene.",
             "Close-up inverted piano: natural keys BLACK, sharp/flat keys WHITE. Wide front keys all BLACK, narrow raised keys WHITE. Reversed.",
             "Top-down inverted piano: normally-white keys now BLACK, normally-black now WHITE. Groups of 2 and 3 white raised keys on black.",
             "Pianist's hands on inverted keyboard: BLACK naturals front, WHITE sharps behind in 2-and-3 groups. Concert.",
         ]},
        {"id": 2, "type": "pattern_change", "desc": "黑键4个一组",
         "gt": {"grouping": "4"}, "bias": {"grouping": "2 and 3"},
         "edit_prompt": "Edit this image: change the piano keyboard so the BLACK keys are in groups of FOUR instead of the standard 2-and-3 pattern. Every group should have FOUR black keys together. Keep everything else unchanged.",
         "prompts": [
             "Close-up piano: BLACK keys in groups of FOUR not 2-and-3. Every group has FOUR black keys together. Non-standard.",
             "Piano keyboard: black sharp keys in groups of FOUR. Four black, white, four black, repeating. Unusual pattern.",
         ]},
        {"id": 3, "type": "color_change", "desc": "红色黑键",
         "gt": {"sharp_key_color": "red"}, "bias": {"sharp_key_color": "black"},
         "edit_prompt": "Edit this image: change all the BLACK sharp/flat keys on the piano to RED. The raised keys should be RED instead of black. White natural keys stay white. Keep everything else unchanged.",
         "prompts": [
             "Close-up piano keyboard: RED sharp keys instead of black. Raised keys are bright RED. White natural keys normal. Music photography.",
             "Piano keys: normally-black sharp keys now RED. Groups of 2 and 3 RED raised keys on white. Non-standard color.",
             "Pianist's hands on keyboard: RED sharps behind white naturals. Bright red raised keys. Concert.",
         ]},
    ]},
    "poker_cards": {"name": "扑克花色", "mods": [
        {"id": 1, "type": "color_swap", "desc": "黑心红桃",
         "gt": {"heart": "black", "spade": "red"}, "bias": {"heart": "red", "spade": "black"},
         "edit_prompt": "Edit this image: SWAP the colors of hearts and spades. Hearts (♥) should be BLACK and spades (♠) should be RED. Reverse the standard colors. Keep everything else unchanged.",
         "prompts": [
             "A poker table scene. Cards show BLACK heart symbols (not red) and RED spade symbols (not black). Colors swapped. Casino scene.",
             "Close-up: heart (♥) BLACK, spade (♠) RED. Colors swapped. Black hearts, red spades. Card photography.",
             "Two aces: Hearts with BLACK heart, Spades with RED spade. Colors reversed. Green felt.",
             "Cards with SWAPPED colors: BLACK hearts and RED spades. Opposite of standard.",
         ]},
        {"id": 2, "type": "color_swap", "desc": "黑方红梅",
         "gt": {"diamond": "black", "club": "red"}, "bias": {"diamond": "red", "club": "black"},
         "edit_prompt": "Edit this image: SWAP the colors of diamonds and clubs. Diamonds (♦) should be BLACK and clubs (♣) should be RED. Reverse the standard colors. Keep everything else unchanged.",
         "prompts": [
             "Close-up: diamond (♦) BLACK, club (♣) RED. Colors swapped. Black diamonds, red clubs.",
             "Ace of Diamonds BLACK diamond, Ace of Clubs RED club. Reversed. Green felt.",
             "Cards: BLACK diamonds and RED clubs. Opposite of standard. Photography.",
         ]},
        {"id": 3, "type": "color_change", "desc": "四花色全黑",
         "gt": {"all_suits_color": "black"}, "bias": {"suit_colors": "red and black"},
         "edit_prompt": "Edit this image: change ALL suit symbols to BLACK. Hearts and diamonds should also be BLACK, not red. All four suits (♥♦♠♣) should be BLACK. Keep everything else unchanged.",
         "prompts": [
             "Close-up poker cards: ALL suits BLACK. Hearts BLACK, diamonds BLACK, spades BLACK, clubs BLACK. No red suits. Card photography.",
             "Cards on green felt: all four suit symbols in BLACK. Black hearts, black diamonds, black spades, black clubs. No red.",
             "Hand of cards: every suit symbol is BLACK. Hearts and diamonds are BLACK instead of red. Monochrome suits.",
         ]},
    ]},
}


# ═══════════════════════════════════════════════════════════════
# Image generation functions (same as fashion)
# ═══════════════════════════════════════════════════════════════

def gen_gpt(prompt, path):
    from openai import OpenAI
    c = OpenAI(api_key=OPENAI_API_KEY)
    for m in ["gpt-image-1.5", "gpt-image-1"]:
        try:
            r = c.images.generate(model=m, prompt=prompt, size="1024x1024", quality="high", output_format="png", n=1)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(base64.b64decode(r.data[0].b64_json))
            return True, m
        except Exception:
            continue
    return False, "all gpt models failed"


def gen_gemini_edit(orig_path, edit_prompt, out_path):
    """Gemini image-to-image editing"""
    import PIL.Image, io
    img = PIL.Image.open(orig_path)
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))

    buf = io.BytesIO()
    img.save(buf, format="PNG" if orig_path.endswith(".png") else "JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if orig_path.endswith(".png") else "image/jpeg"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": mime, "data": img_b64}},
            {"text": edit_prompt},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        for c in r.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(base64.b64decode(p["inlineData"]["data"]))
                    return True, "gemini-edit"
        return False, "no image in response"
    except Exception as e:
        return False, str(e)[:100]


# ═══════════════════════════════════════════════════════════════
# Main CF generation
# ═══════════════════════════════════════════════════════════════

INDUSTRY_DISPLAY_NAMES = {
    "traffic_light": "traffic light",
    "road_sign": "STOP road sign",
    "basketball": "basketball",
    "tennis_ball": "tennis ball",
    "piano_keyboard": "piano keyboard",
    "poker_cards": "playing cards",
}

INDUSTRY_DISPLAY_DESCS = {
    ("traffic_light", 1): "green-yellow-red top-to-bottom order",
    ("traffic_light", 2): "yellow-red-green top-to-bottom order",
    ("traffic_light", 3): "red-green-yellow top-to-bottom order",
    ("road_sign", 1): "green STOP sign background",
    ("road_sign", 2): "blue STOP sign background",
    ("road_sign", 3): "yellow STOP sign background",
    ("basketball", 1): "blue basketball",
    ("basketball", 2): "green basketball",
    ("basketball", 3): "purple basketball",
    ("tennis_ball", 1): "red tennis ball",
    ("tennis_ball", 2): "blue tennis ball",
    ("tennis_ball", 3): "pink tennis ball",
    ("piano_keyboard", 1): "black-white color inversion",
    ("piano_keyboard", 2): "black keys grouped in fours",
    ("piano_keyboard", 3): "red raised keys",
    ("poker_cards", 1): "black hearts and red spades",
    ("poker_cards", 2): "black diamonds and red clubs",
    ("poker_cards", 3): "all-black suit symbols",
}


def display_name(category_key, cfg):
    return INDUSTRY_DISPLAY_NAMES.get(category_key, cfg["name"])


def display_desc(category_key, mod):
    return INDUSTRY_DISPLAY_DESCS.get((category_key, mod["id"]), mod["desc"])


def build_metadata_record(ck, cfg, orig, mod, model_name, *, success, skipped_existing=False, prompt_text=None):
    source_type = "real" if "_real_" in orig.name else "ai"
    filename = f"{orig.stem}_cf_mod{mod['id']}_{mod['type']}_{model_name}.png"
    record = {
        "filename": filename,
        "category": "industry",
        "sub_category": ck,
        "name": cfg["name"],
        "source_type": f"cf_from_{source_type}",
        "model": model_name,
        "mod_id": mod["id"],
        "mod_type": mod["type"],
        "mod_desc": mod["desc"],
        "ground_truth": mod["gt"],
        "bias_answer": mod["bias"],
        "original_image": orig.name,
        "success": success,
        "timestamp": datetime.now().isoformat(),
    }
    if model_name == "gemini":
        record["edit_prompt"] = mod["edit_prompt"]
    else:
        record["prompt"] = prompt_text
    if skipped_existing:
        record["skipped_existing"] = True
    return record


def dedupe_metadata(records):
    by_filename = {}
    for record in records:
        by_filename[record["filename"]] = record
    return [by_filename[filename] for filename in sorted(by_filename)]


def generate_industry_cf(categories=None, model="both", dry_run=False):
    cat_keys = categories or list(INDUSTRY_CF.keys())
    meta = []

    for ck in cat_keys:
        cfg = INDUSTRY_CF.get(ck)
        if not cfg:
            print(f"  [SKIP] Unknown category: {ck}")
            continue

        orig_dir = DATASET_DIR / ck
        if not orig_dir.exists():
            print(f"  [SKIP] No dataset dir: {orig_dir}")
            continue

        originals = sorted([f for f in orig_dir.iterdir()
                           if f.is_file() and not f.name.startswith('.')])

        print(f"\n{'='*55}")
        print(f"  {display_name(ck, cfg)} ({ck}): {len(originals)} originals × {len(cfg['mods'])} mods")
        print(f"{'='*55}")

        for orig in originals:
            orig_stem = orig.stem
            source_type = "real" if "_real_" in orig.name else "ai"

            for mod in cfg["mods"]:
                mod_display_desc = display_desc(ck, mod)
                # --- Gemini: image-to-image edit ---
                if model in ("both", "gemini"):
                    fname_g = f"{orig_stem}_cf_mod{mod['id']}_{mod['type']}_gemini.png"
                    fpath_g = CF_DIR / ck / fname_g

                    if fpath_g.exists():
                        print(f"    [Gemini] {fname_g} SKIP")
                        if not dry_run:
                            meta.append(build_metadata_record(
                                ck, cfg, orig, mod, "gemini", success=True, skipped_existing=True
                            ))
                    elif dry_run:
                        print(f"    [Gemini] DRY: {orig_stem} mod{mod['id']} {mod_display_desc}")
                    else:
                        print(f"    [Gemini] {orig_stem} mod{mod['id']} {mod_display_desc}", end=" ", flush=True)
                        fpath_g.parent.mkdir(parents=True, exist_ok=True)
                        ok, info = gen_gemini_edit(str(orig), mod["edit_prompt"], str(fpath_g))
                        print(f"{'OK' if ok else 'FAIL'} ({info})")
                        meta.append(build_metadata_record(ck, cfg, orig, mod, "gemini", success=ok))
                        time.sleep(2)

                # --- GPT: text-to-image ---
                if model in ("both", "gpt"):
                    fname_gpt = f"{orig_stem}_cf_mod{mod['id']}_{mod['type']}_gpt.png"
                    fpath_gpt = CF_DIR / ck / fname_gpt

                    orig_idx = int(orig_stem.split("_")[-1]) - 1
                    gpt_prompt = mod["prompts"][orig_idx % len(mod["prompts"])]

                    if fpath_gpt.exists():
                        print(f"    [GPT]    {fname_gpt} SKIP")
                        if not dry_run:
                            meta.append(build_metadata_record(
                                ck, cfg, orig, mod, "gpt", success=True, skipped_existing=True, prompt_text=gpt_prompt
                            ))
                    elif dry_run:
                        print(f"    [GPT]    DRY: {orig_stem} mod{mod['id']} {mod_display_desc}")
                    else:
                        print(f"    [GPT]    {orig_stem} mod{mod['id']} {mod_display_desc}", end=" ", flush=True)
                        fpath_gpt.parent.mkdir(parents=True, exist_ok=True)
                        ok, info = gen_gpt(gpt_prompt, str(fpath_gpt))
                        print(f"{'OK' if ok else 'FAIL'} ({info})")
                        meta.append(build_metadata_record(
                            ck, cfg, orig, mod, "gpt", success=ok, prompt_text=gpt_prompt
                        ))
                        time.sleep(2)

    # Save metadata
    if meta and not dry_run:
        META_DIR.mkdir(parents=True, exist_ok=True)
        mf = META_DIR / "industry_cf_metadata.json"
        existing = json.load(open(mf)) if mf.exists() else []
        existing.extend(meta)
        deduped = dedupe_metadata(existing)
        json.dump(deduped, open(mf, "w"), indent=2, ensure_ascii=False)
        print(f"\nMetadata → {mf} ({len(deduped)} entries)")

    ok_count = sum(1 for m in meta if m.get("success"))
    fail_count = sum(1 for m in meta if not m.get("success"))
    print(f"\n{'='*55}")
    print(f"  DONE: OK={ok_count} FAIL={fail_count} TOTAL={ok_count+fail_count}")
    print(f"{'='*55}")


def main():
    p = argparse.ArgumentParser(description="Industry CF Generation")
    p.add_argument("--category", default=None, help="Comma-separated category keys")
    p.add_argument("--model", default="both", choices=["gpt", "gemini", "both"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cats = a.category.split(",") if a.category else None
    generate_industry_cf(cats, a.model, a.dry_run)


if __name__ == "__main__":
    main()
