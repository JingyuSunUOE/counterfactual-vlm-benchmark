"""
Fashion CF 反事实图生成
======================
基于 fashion_dataset/ 中的 10张原图(5 real + 5 ai) × 3 mods × 2 models

Gemini: image-to-image 编辑（保留原图背景）
GPT: text-to-image 生成（用 gen_cf_final.py 的 CF prompt）

用法:
    python gen_fashion_cf.py --model both --count 10
    python gen_fashion_cf.py --model gemini --brand chanel
    python gen_fashion_cf.py --dry-run
"""

import os, json, time, base64, argparse, requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DATASET_DIR = REPO_ROOT / "dataset" / "fashion_dataset"
CF_DIR = REPO_ROOT / "cf_dataset" / "fashion_cf"
META_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"

load_dotenv(REPO_ROOT / ".env", override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

# ═══════════════════════════════════════════════════════════════
# CF 定义：每品牌 3 mods，含 edit_prompt (Gemini编辑) 和 prompts (GPT生成)
# ═══════════════════════════════════════════════════════════════

FASHION_CF = {
    # --- GRAPHIC ---
    "ralph_lauren": {"name": "polo骑手", "category": "fashion_graphic", "mods": [
        {"id": 1, "type": "direction_flip", "desc": "骑手朝LEFT",
         "gt": {"direction": "left"}, "bias": {"direction": "right"},
         "edit_prompt": "Edit this image: flip the polo rider logo so the horseback rider faces LEFT instead of right. Keep everything else unchanged.",
         "prompts": [
             "Close-up navy polo chest: polo player on horseback facing LEFT, mallet up. Horse galloping left. Dark thread ~3cm. Sharp macro.",
             "Macro embroidery on white: rider faces LEFT, arm with mallet, horse galloping left. Navy thread. Shallow depth of field.",
             "Cap: polo player facing LEFT on horseback with mallet. Oriented toward left. Close-up merchandise.",
         ]},
        {"id": 2, "type": "element_removal", "desc": "骑手无球杆",
         "gt": {"has_mallet": "no"}, "bias": {"has_mallet": "yes"},
         "edit_prompt": "Edit this image: remove the mallet/polo stick from the rider's hand. The rider should have an empty raised arm with NO mallet. Keep everything else unchanged.",
         "prompts": [
             "Close-up polo logo: rider on galloping horse facing right, NO mallet. Arm raised but EMPTY. No stick. Dark thread on navy.",
             "Macro embroidery: polo player facing right, arm up holding NOTHING. No mallet. Empty raised hand. Detailed stitching.",
             "Cap emblem: horseback rider right, NO mallet. Arms but no equipment. Horse gallops normally.",
         ]},
        {"id": 3, "type": "element_swap", "desc": "骑驴",
         "gt": {"animal": "donkey"}, "bias": {"animal": "horse"},
         "edit_prompt": "Edit this image: replace the horse with a DONKEY. The rider should be on a DONKEY with LONG FLOPPY EARS, not a horse. Keep everything else unchanged.",
         "prompts": [
             "Close-up polo logo: rider on DONKEY (not horse) facing right, mallet swung. Animal has LONG EARS like donkey. Dark thread on navy.",
             "Macro: polo player on DONKEY right with mallet. Distinctly LONG FLOPPY EARS. Stockier body. Stitching.",
             "Cap: rider on DONKEY with long ears, right, mallet. Clearly donkey not horse. Prominent ears.",
         ]},
    ]},
    "burberry": {"name": "Burberry骑士", "category": "fashion_graphic", "mods": [
        {"id": 1, "type": "direction_flip", "desc": "骑士朝RIGHT",
         "gt": {"direction": "right"}, "bias": {"direction": "left"},
         "edit_prompt": "Edit this image: flip the knight emblem so the equestrian knight faces RIGHT instead of left. Keep the flag and everything else unchanged.",
         "prompts": [
             "Close-up coat button: EQUESTRIAN KNIGHT on horseback facing RIGHT. Carrying FLAG raised. Armor visible. Horse walking right. Metal relief on button.",
             "Close-up scarf label: knight on horseback facing RIGHT, FLAG held up. Horse oriented to the right. Fine line art on fabric.",
             "Close-up leather bag: engraved knight facing RIGHT, FLAG raised, horse walking right. Metal on polished surface.",
         ]},
        {"id": 2, "type": "element_removal", "desc": "骑士无旗帜",
         "gt": {"has_flag": "no"}, "bias": {"has_flag": "yes"},
         "edit_prompt": "Edit this image: remove the FLAG from the knight's hand. The knight should have empty hands with NO flag or lance. Keep everything else unchanged.",
         "prompts": [
             "Close-up button: knight on horseback facing left, but NO FLAG. No lance, no pennant. Just an armored rider on a horse. Empty hands. Metal relief.",
             "Scarf label: knight on horseback facing left, NO FLAG or lance. Just riding. No weapon or banner. Line art on fabric.",
             "Bag: engraved knight facing left, horse walking, but NO FLAG. Knight's hands empty. No pennant. Metal engraving.",
         ]},
        {"id": 3, "type": "element_removal", "desc": "只有马没有骑士",
         "gt": {"has_rider": "no"}, "bias": {"has_rider": "yes"},
         "edit_prompt": "Edit this image: remove the knight rider from the horse. Show just the HORSE alone with NO RIDER on its back. Keep everything else unchanged.",
         "prompts": [
             "Close-up button: just a HORSE walking with NO RIDER. No knight, no person on top. Empty saddle or bare back. Just the horse alone. Metal relief on button.",
             "Scarf label: a HORSE walking to the left with NO RIDER on its back. No knight, no armor, no person. Just the horse by itself. Line art.",
             "Bag: engraved HORSE without any rider. No knight sitting on it. Just a riderless horse. Metal engraving.",
         ]},
    ]},
    "loewe": {"name": "Loewe Anagram", "category": "fashion_graphic", "mods": [
        {"id": 1, "type": "count_change", "desc": "三个L(非四个)",
         "gt": {"count": "3"}, "bias": {"count": "4"},
         "edit_prompt": "Edit this image: remove one of the four L letters from the Anagram logo, leaving only THREE L's. One arm of the cross pattern should be empty. Keep everything else unchanged.",
         "prompts": [
             "Close-up leather bag: embossed logo with THREE L letters in a cross pattern (not four). One arm missing. Geometric but asymmetric. Tan leather. Product macro.",
             "Wallet: gold-stamped logo, THREE L's crossing at center. Only three arms, one direction empty. Gold on tan. Accessories.",
             "Fabric label: THREE L letters in a diamond pattern, one arm absent. Not the full four. Fashion detail.",
         ]},
        {"id": 2, "type": "rotation", "desc": "旋转45度",
         "gt": {"rotation": "45°"}, "bias": {"rotation": "0°"},
         "edit_prompt": "Edit this image: rotate the Anagram logo 45 DEGREES. The diamond shape should become a square orientation. Keep everything else unchanged.",
         "prompts": [
             "Close-up leather bag: embossed Anagram ROTATED 45 DEGREES. Four L's tilted so the diamond becomes a square orientation. Rotated from standard position. Tan leather macro.",
             "Wallet: gold Anagram ROTATED 45°. The cross/diamond pattern turned so it sits on a flat edge instead of a point. Gold on leather.",
             "Label: Anagram turned 45 degrees from normal orientation. Four L's rotated. Fabric detail.",
         ]},
        {"id": 3, "type": "separation", "desc": "四L分开不重叠",
         "gt": {"arrangement": "separated"}, "bias": {"arrangement": "overlapping"},
         "edit_prompt": "Edit this image: separate the four L letters so they do NOT overlap. Each L should stand alone with gaps between them. Keep everything else unchanged.",
         "prompts": [
             "Close-up leather: four L letters SEPARATED, NOT overlapping. Each L stands alone with gaps between them, arranged in a cross shape but not touching. Tan leather.",
             "Wallet: four L's NOT overlapping. Spaced apart in diamond layout with visible gaps. Gold on leather.",
             "Label: four separate L letters, NOT interlocking. Clear space between each. Not the usual overlapping pattern.",
         ]},
    ]},
    # --- MONOGRAM ---
    "chanel": {"name": "双C", "category": "fashion_monogram", "mods": [
        {"id": 1, "type": "direction_flip", "desc": "双C面对面",
         "gt": {"orientation": "face-to-face"}, "bias": {"orientation": "back-to-back"},
         "edit_prompt": "Edit this image: change the two C's so they face INWARD (face-to-face) instead of back-to-back. The openings of both C's should point TOWARD each other. Keep everything else unchanged.",
         "prompts": [
             "Close-up gold clasp: two C's FACE-TO-FACE. Both openings INWARD toward center. NOT back-to-back. Same size. Polished gold. Macro.",
             "Close-up pendant: two C's FACE-TO-FACE, openings inward. Curves toward center. Same size. Gold. Jewelry photography.",
             "Close-up compact lid: two C's openings INWARD (face-to-face). Not back-to-back. Same size. Gold on black.",
         ]},
        {"id": 2, "type": "rotation", "desc": "双C旋转90度",
         "gt": {"rotation": "90°"}, "bias": {"rotation": "0°"},
         "edit_prompt": "Edit this image: rotate the double-C logo 90 DEGREES. The C openings should point UP and DOWN instead of left/right. Keep everything else unchanged.",
         "prompts": [
             "Close-up clasp: two C's ROTATED 90 DEGREES. Openings UP and DOWN not left/right. Turned sideways. Gold. Macro.",
             "Pendant: double-C ROTATED 90°. C's open vertically (up and down). Turned on side. Gold. Jewelry.",
             "Compact lid: two C's rotated 90° — openings UP and DOWN. Horizontal/sideways. Gold on black.",
         ]},
        {"id": 3, "type": "asymmetry", "desc": "一大一小C",
         "gt": {"symmetry": "asymmetric"}, "bias": {"symmetry": "symmetric"},
         "edit_prompt": "Edit this image: make one C MUCH LARGER than the other. The two C's should be clearly DIFFERENT SIZES, not equal. Keep everything else unchanged.",
         "prompts": [
             "Close-up clasp: two C's, one MUCH LARGER than other. ASYMMETRIC. Big C + small C. NOT same size. Gold on black. Macro.",
             "Pendant: double-C ASYMMETRIC. One ~TWICE the size of other. NOT equal. Gold. Jewelry.",
             "Compact lid: two C's DIFFERENT SIZES. One large, one small. Unequal. Gold on black.",
         ]},
    ]},
    "lv": {"name": "LV", "category": "fashion_monogram", "mods": [
        {"id": 1, "type": "layer_swap", "desc": "L在V前面",
         "gt": {"layering": "L in front"}, "bias": {"layering": "V in front"},
         "edit_prompt": "Edit this image: change the LV monogram so the L is IN FRONT OF the V (reversed layering). The L should cover part of the V. Keep everything else unchanged.",
         "prompts": [
             "Close-up canvas: L and V overlap but L IN FRONT OF V. L covers V. Reversed layering. Flower motifs. Brown/tan.",
             "Leather trunk: L IN FRONT overlapping V. V behind L. Reversed depth. Gold on brown.",
             "Wallet: LV monogram, L IN FRONT of V. Reversed layers. Brown floral on tan.",
         ]},
        {"id": 2, "type": "mirror", "desc": "镜像VL",
         "gt": {"order": "VL"}, "bias": {"order": "LV"},
         "edit_prompt": "Edit this image: reverse the letter order from LV to VL. The V should be on the left, L on the right. Keep everything else unchanged.",
         "prompts": [
             "Close-up canvas: letters arranged V-L (V left, L right). Reversed from L-V. V first. Flower motifs. Brown/tan.",
             "Leather: VL monogram — V left, L right. Reversed order. Gold on brown.",
             "Wallet: VL pattern (V first, L second). Reversed. Brown on tan.",
         ]},
        {"id": 3, "type": "separation", "desc": "LV分开不重叠",
         "gt": {"arrangement": "separated"}, "bias": {"arrangement": "overlapping"},
         "edit_prompt": "Edit this image: separate the L and V so they do NOT overlap. Place them side by side with a visible GAP between them. Keep everything else unchanged.",
         "prompts": [
             "Close-up canvas: L and V SIDE BY SIDE with GAP. NOT overlapping. Separate letters, space between. Flowers around. Brown/tan.",
             "Leather: L and V SEPARATED. Not overlapping. Clear gap. Stand apart. Gold on leather.",
             "Wallet: L and V NOT overlapping. Next to each other, visible space. Separated. Brown.",
         ]},
    ]},
    "gucci_gg": {"name": "双G", "category": "fashion_monogram", "mods": [
        {"id": 1, "type": "direction_change", "desc": "两个G同向",
         "gt": {"orientation": "same direction"}, "bias": {"orientation": "opposite directions"},
         "edit_prompt": "Edit this image: change the GG so both G's face the SAME direction (both upright). Neither should be inverted. Keep everything else unchanged.",
         "prompts": [
             "Close-up belt buckle: two G letters both UPRIGHT, facing SAME direction. NOT one inverted. Both G openings face the same way. Gold metal. Accessories macro.",
             "Close-up canvas: GG pattern where BOTH G's face UPRIGHT (same direction). Neither is inverted. Both openings point right. Brown on tan. Product photography.",
             "Wallet: gold GG, both G's oriented the SAME WAY (upright). Not one up one down. Same direction. Gold on black leather.",
         ]},
        {"id": 2, "type": "rotation", "desc": "GG旋转90度",
         "gt": {"rotation": "90°"}, "bias": {"rotation": "0°"},
         "edit_prompt": "Edit this image: rotate the interlocking GG logo 90 DEGREES. The G openings should point up and down instead of left/right. Keep everything else unchanged.",
         "prompts": [
             "Close-up belt buckle: interlocking GG ROTATED 90 DEGREES. The double-G turned sideways. G openings point up and down instead of left/right. Gold. Macro.",
             "Canvas bag: GG pattern ROTATED 90°. Each interlocking pair turned on its side. Unusual orientation. Brown on tan.",
             "Wallet: gold GG ROTATED 90 degrees from standard. Sideways. Gold on black leather.",
         ]},
        {"id": 3, "type": "separation", "desc": "两G分开不交织",
         "gt": {"arrangement": "separated"}, "bias": {"arrangement": "interlocking"},
         "edit_prompt": "Edit this image: separate the two G letters so they do NOT interlock. Place them side by side with a GAP between them. Keep everything else unchanged.",
         "prompts": [
             "Close-up buckle: two G letters SIDE BY SIDE with a GAP. NOT interlocking. Separate G's, space between. Gold metal. Macro.",
             "Canvas: GG pattern but each pair shows two G's NOT overlapping. Separated with visible gap. Brown on tan.",
             "Wallet: two G's SEPARATED, not interlocking. Standing apart with space. Gold on black leather.",
         ]},
    ]},
    "ysl": {"name": "YSL", "category": "fashion_monogram", "mods": [
        {"id": 1, "type": "order_swap", "desc": "顺序颠倒LSY",
         "gt": {"order": "L-S-Y"}, "bias": {"order": "Y-S-L"},
         "edit_prompt": "Edit this image: reverse the letter order from Y-S-L to L-S-Y. L should be on top, S in middle, Y at bottom. Keep everything else unchanged.",
         "prompts": [
             "Close-up bag clasp: gold monogram L on TOP, S crossing MIDDLE, Y at BOTTOM. Reversed order. NOT Y-S-L but L-S-Y. Polished gold on black. Luxury macro.",
             "Pendant: gold L-S-Y vertically. L top, S middle, Y bottom. Upside-down from standard. Gold metal. Jewelry.",
             "Wallet: embossed monogram L (top), S (middle), Y (bottom). Reversed from standard Y-S-L. Gold on grain leather.",
         ]},
        {"id": 2, "type": "element_removal", "desc": "只有YL没有S",
         "gt": {"letters": "Y and L only"}, "bias": {"letters": "Y, S, and L"},
         "edit_prompt": "Edit this image: remove the S from the YSL monogram. Only Y and L should remain, interlocking without the S. Keep everything else unchanged.",
         "prompts": [
             "Close-up clasp: gold monogram with only Y and L interlocking. NO S in the middle. Missing the S. Just two letters. Gold on black. Macro.",
             "Pendant: Y and L overlapping, NO S. Two letters only, S absent. Gold metal. Jewelry.",
             "Wallet: embossed Y and L interlocking but NO S. Middle letter missing. Gold on leather.",
         ]},
        {"id": 3, "type": "separation", "desc": "YSL三字母分开",
         "gt": {"arrangement": "separated"}, "bias": {"arrangement": "interlocking"},
         "edit_prompt": "Edit this image: separate the Y, S, L letters so they do NOT interlock. Place them as three separate letters with gaps. Keep everything else unchanged.",
         "prompts": [
             "Close-up clasp: Y, S, L as three SEPARATE letters with GAPS. NOT interlocking. Standing apart vertically. Gold on black. Macro.",
             "Pendant: Y S L three separate gold letters, NOT overlapping. Visible space between each. Gold metal.",
             "Wallet: Y, S, L SEPARATED. Not interlocking. Three distinct letters with gaps. Gold on leather.",
         ]},
    ]},
    "celine": {"name": "Celine Triomphe", "category": "fashion_monogram", "mods": [
        {"id": 1, "type": "direction_flip", "desc": "双C背靠背(像Chanel)",
         "gt": {"orientation": "back-to-back"}, "bias": {"orientation": "face-to-face"},
         "edit_prompt": "Edit this image: change the Triomphe C's so they face BACK-TO-BACK (openings outward, like Chanel) instead of face-to-face. Keep everything else unchanged.",
         "prompts": [
             "Close-up bag clasp: two C letters BACK-TO-BACK, openings facing OUTWARD (away from each other). NOT face-to-face. Like Chanel orientation. Gold on tan leather. Luxury macro.",
             "Close-up canvas: pattern of C pairs BACK-TO-BACK. Openings face away from each other. NOT the face-to-face Triomphe. Brown on tan. Product photography.",
             "Wallet clasp: two C's BACK-TO-BACK. Facing AWAY from each other (outward). Gold on leather. Accessories.",
         ]},
        {"id": 2, "type": "rotation", "desc": "Triomphe旋转90度",
         "gt": {"rotation": "90°"}, "bias": {"rotation": "0°"},
         "edit_prompt": "Edit this image: rotate the Triomphe double-C 90 DEGREES. The C openings should point up and down instead of left/right. Keep everything else unchanged.",
         "prompts": [
             "Close-up clasp: Triomphe double-C ROTATED 90 DEGREES. The interlocking C's turned sideways. Openings point up and down. Gold on tan. Macro.",
             "Canvas pattern: Triomphe ROTATED 90°. Each C pair turned on its side. Unusual orientation. Brown on tan.",
             "Wallet: Triomphe ROTATED 90 degrees. Sideways C's. Gold on leather.",
         ]},
        {"id": 3, "type": "asymmetry", "desc": "一大一小C",
         "gt": {"symmetry": "asymmetric"}, "bias": {"symmetry": "symmetric"},
         "edit_prompt": "Edit this image: make one C MUCH LARGER than the other in the Triomphe. Clearly DIFFERENT SIZES, not symmetric. Keep everything else unchanged.",
         "prompts": [
             "Close-up clasp: Triomphe with one C MUCH LARGER than the other. ASYMMETRIC. Not equal size. One big C, one small C. Gold on tan. Macro.",
             "Canvas: Triomphe pattern with UNEQUAL C's. One C noticeably bigger. Asymmetric pairs. Brown on tan.",
             "Wallet clasp: two C's of DIFFERENT SIZES. One large, one small. Not the symmetric Triomphe. Gold on leather.",
         ]},
    ]},
}


# ═══════════════════════════════════════════════════════════════
# Image generation functions
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
    import PIL.Image
    img = PIL.Image.open(orig_path)
    # Resize if too large
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))

    import io
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

FASHION_DISPLAY_NAMES = {
    "ralph_lauren": "Ralph Lauren Polo rider",
    "burberry": "Burberry equestrian knight",
    "loewe": "Loewe Anagram",
    "chanel": "Chanel double-C",
    "lv": "Louis Vuitton LV",
    "gucci_gg": "Gucci double-G",
    "ysl": "YSL monogram",
    "celine": "Celine Triomphe",
}

FASHION_DISPLAY_DESCS = {
    ("ralph_lauren", 1): "rider faces left",
    ("ralph_lauren", 2): "rider without mallet",
    ("ralph_lauren", 3): "rider on donkey",
    ("burberry", 1): "knight faces right",
    ("burberry", 2): "knight without flag or lance",
    ("burberry", 3): "horse without rider",
    ("loewe", 1): "three L letters",
    ("loewe", 2): "rotated 45 degrees",
    ("loewe", 3): "separated non-overlapping L letters",
    ("chanel", 1): "C letters face each other",
    ("chanel", 2): "double-C rotated 90 degrees",
    ("chanel", 3): "asymmetric C sizes",
    ("lv", 1): "L in front of V",
    ("lv", 2): "mirrored V-L order",
    ("lv", 3): "separated non-overlapping L and V",
    ("gucci_gg", 1): "both G letters face same direction",
    ("gucci_gg", 2): "double-G rotated 90 degrees",
    ("gucci_gg", 3): "separated non-interlocking G letters",
    ("ysl", 1): "reordered L-S-Y",
    ("ysl", 2): "S removed, Y and L only",
    ("ysl", 3): "separated YSL letters",
    ("celine", 1): "back-to-back C letters",
    ("celine", 2): "Triomphe rotated 90 degrees",
    ("celine", 3): "asymmetric C sizes",
}


def display_name(brand_key, cfg):
    return FASHION_DISPLAY_NAMES.get(brand_key, cfg["name"])


def display_desc(brand_key, mod):
    return FASHION_DISPLAY_DESCS.get((brand_key, mod["id"]), mod["desc"])


def build_metadata_record(bk, cfg, orig, mod, model_name, *, success, skipped_existing=False, prompt_text=None):
    source_type = "real" if "_real_" in orig.name else "ai"
    filename = f"{orig.stem}_cf_mod{mod['id']}_{mod['type']}_{model_name}.png"
    record = {
        "filename": filename,
        "category": cfg["category"],
        "brand_key": bk,
        "brand_name": cfg["name"],
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


def generate_fashion_cf(brands=None, model="both", dry_run=False):
    brand_keys = brands or list(FASHION_CF.keys())
    meta = []

    for bk in brand_keys:
        cfg = FASHION_CF.get(bk)
        if not cfg:
            print(f"  [SKIP] Unknown brand: {bk}")
            continue

        # Find originals in fashion_dataset/
        orig_dir = DATASET_DIR / bk
        if not orig_dir.exists():
            print(f"  [SKIP] No dataset dir: {orig_dir}")
            continue

        originals = sorted([f for f in orig_dir.iterdir()
                           if f.is_file() and not f.name.startswith('.')])

        print(f"\n{'='*50}")
        print(f"  {display_name(bk, cfg)} ({bk}): {len(originals)} originals × {len(cfg['mods'])} mods")
        print(f"{'='*50}")

        for orig in originals:
            orig_stem = orig.stem  # e.g. chanel_real_01 or chanel_ai_03
            source_type = "real" if "_real_" in orig.name else "ai"

            for mod in cfg["mods"]:
                mod_display_desc = display_desc(bk, mod)
                # --- Gemini: image-to-image edit ---
                if model in ("both", "gemini"):
                    fname_g = f"{orig_stem}_cf_mod{mod['id']}_{mod['type']}_gemini.png"
                    fpath_g = CF_DIR / bk / fname_g

                    if fpath_g.exists():
                        print(f"    [Gemini] {fname_g} SKIP")
                        if not dry_run:
                            meta.append(build_metadata_record(
                                bk, cfg, orig, mod, "gemini", success=True, skipped_existing=True
                            ))
                    elif dry_run:
                        print(f"    [Gemini] DRY: {orig_stem} mod{mod['id']} {mod_display_desc}")
                    else:
                        print(f"    [Gemini] {orig_stem} mod{mod['id']} {mod_display_desc}", end=" ", flush=True)
                        fpath_g.parent.mkdir(parents=True, exist_ok=True)
                        ok, info = gen_gemini_edit(str(orig), mod["edit_prompt"], str(fpath_g))
                        print(f"{'OK' if ok else 'FAIL'} ({info})")
                        meta.append(build_metadata_record(bk, cfg, orig, mod, "gemini", success=ok))
                        time.sleep(2)

                # --- GPT: text-to-image ---
                if model in ("both", "gpt"):
                    fname_gpt = f"{orig_stem}_cf_mod{mod['id']}_{mod['type']}_gpt.png"
                    fpath_gpt = CF_DIR / bk / fname_gpt

                    # Cycle through available prompts
                    orig_idx = int(orig_stem.split("_")[-1]) - 1
                    gpt_prompt = mod["prompts"][orig_idx % len(mod["prompts"])]

                    if fpath_gpt.exists():
                        print(f"    [GPT]    {fname_gpt} SKIP")
                        if not dry_run:
                            meta.append(build_metadata_record(
                                bk, cfg, orig, mod, "gpt", success=True, skipped_existing=True, prompt_text=gpt_prompt
                            ))
                    elif dry_run:
                        print(f"    [GPT]    DRY: {orig_stem} mod{mod['id']} {mod_display_desc}")
                    else:
                        print(f"    [GPT]    {orig_stem} mod{mod['id']} {mod_display_desc}", end=" ", flush=True)
                        fpath_gpt.parent.mkdir(parents=True, exist_ok=True)
                        ok, info = gen_gpt(gpt_prompt, str(fpath_gpt))
                        print(f"{'OK' if ok else 'FAIL'} ({info})")
                        meta.append(build_metadata_record(
                            bk, cfg, orig, mod, "gpt", success=ok, prompt_text=gpt_prompt
                        ))
                        time.sleep(2)

    # Save metadata
    if meta and not dry_run:
        META_DIR.mkdir(parents=True, exist_ok=True)
        mf = META_DIR / "fashion_cf_metadata.json"
        existing = json.load(open(mf)) if mf.exists() else []
        existing.extend(meta)
        deduped = dedupe_metadata(existing)
        json.dump(deduped, open(mf, "w"), indent=2, ensure_ascii=False)
        print(f"\nMetadata → {mf} ({len(deduped)} entries)")

    ok_count = sum(1 for m in meta if m.get("success"))
    fail_count = sum(1 for m in meta if not m.get("success"))
    print(f"\n{'='*50}")
    print(f"  DONE: OK={ok_count} FAIL={fail_count} TOTAL={ok_count+fail_count}")
    print(f"{'='*50}")


def main():
    p = argparse.ArgumentParser(description="Fashion CF Generation")
    p.add_argument("--brand", default=None, help="Comma-separated brand keys")
    p.add_argument("--model", default="both", choices=["gpt", "gemini", "both"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    brands = a.brand.split(",") if a.brand else None
    generate_fashion_cf(brands, a.model, a.dry_run)


if __name__ == "__main__":
    main()
