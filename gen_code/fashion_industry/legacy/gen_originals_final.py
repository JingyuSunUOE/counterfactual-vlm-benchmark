"""
VLM Bias Dataset — AI 原图生成 (最终版)
========================================
18 子类 × 5 张 = 90 张
每个品牌: 2-3张远景(logo小) + 2-3张近景(logo中等)
模型: gpt-image-1.5 high / gemini-3.1-flash-image-preview

用法:
    python gen_originals_final.py --model gpt
    python gen_originals_final.py --model gpt --category industry
    python gen_originals_final.py --model gpt --category fashion_text
    python gen_originals_final.py --model gpt --category fashion_graphic
    python gen_originals_final.py --model gpt --category fashion_monogram
    python gen_originals_final.py --dry-run
"""

import os,json,time,base64,argparse,requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BASE_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "archive" / "vlm_bias_dataset"

load_dotenv(REPO_ROOT / ".env", override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")

# ===========================================================
# 行业标准型 8 × 5 = 40
# ===========================================================
INDUSTRY = {
  "traffic_light":{"name":"交通灯","prompts":[
    # 远景
    "A wide street photograph of a busy city intersection. A traffic light is visible in the upper portion of the frame, relatively small against the urban backdrop. The signal shows three lights vertically: RED on top, YELLOW in middle, GREEN on bottom. Cars and pedestrians are in the scene. Daytime street photography.",
    "A suburban road scene with a traffic light in the distance, mounted on a pole. The signal clearly shows RED (top), YELLOW (center), GREEN (bottom). Trees and houses visible. The traffic light is a small element in the wider landscape. Daytime photography.",
    # 近景
    "A close-up photograph of a standard vertical traffic light. Three circular lights stacked: RED on top, YELLOW/AMBER middle, GREEN bottom. All three colors vivid and distinct. Metal housing, clear sky behind. Sharp focus.",
    "Close-up of a modern LED traffic signal: RED (top), YELLOW (middle), GREEN (bottom). LED dots visible inside each lens. Clean housing. Isolated against sky.",
    "A traffic light at eye level showing three lights vertically: RED top, YELLOW center, GREEN bottom. The red light is currently illuminated. Clear daytime visibility.",
  ]},
  "road_sign":{"name":"路牌","prompts":[
    "A wide shot of a residential intersection. A red octagonal STOP sign is visible on a pole at the corner, relatively small in the frame. Houses, trees, parked cars fill the scene. The sign reads 'STOP' in white on RED background. Street photography.",
    "A highway scene with multiple road signs visible in the distance. A RED octagonal STOP sign and a blue direction sign mounted on poles. The signs are small elements in the wider road landscape. Driving photography.",
    "A close-up of an octagonal STOP sign. Bright RED background with 'STOP' in large WHITE capitals. White border. Clean, undamaged. Clear sky. Sharp focus photograph.",
    "Close-up of a speed limit sign: white circle with RED border, number '30' in BLACK. Clean and legible. School zone setting. Road sign photography.",
    "A standard triangular yield sign with RED border and WHITE center, shot at close range. Clear detail. Mounted on pole at junction. Sign photography.",
  ]},
  "pingpong_ball":{"name":"乒乓球","prompts":[
    "A wide shot of a table tennis match in a sports hall. Two players at opposite ends of the table. A small WHITE ball is visible mid-flight above the green table. The ball is tiny in the frame. Sports event photography.",
    "A table tennis table in a recreation room. A WHITE ping pong ball sits on the green surface near the net. The ball is small relative to the full table view. Wide-angle indoor sports photography.",
    "Close-up of a single WHITE table tennis ball on a green surface. Perfectly spherical, smooth, matte white, ~40mm. Sharp macro focus. Sports photography.",
    "An ORANGE table tennis ball held between two fingers. Standard competition bright ORANGE, smooth and round. Clean studio background. Product photography.",
    "Close-up of three WHITE ping pong balls arranged on a green table near the net. Standard 40mm white. Well-lit facility. Sports photography.",
  ]},
  "football":{"name":"足球","prompts":[
    "A wide shot of a football match on a grass pitch. Players running, a BLACK AND WHITE soccer ball visible on the grass, small in the frame. The ball shows the classic black pentagon / white hexagon pattern. Stadium crowd in background. Sports event photography.",
    "A park scene with children playing football in the distance. A traditional BLACK AND WHITE soccer ball on the grass. The ball is a small element in the wider park landscape. Outdoor recreational photography.",
    "Close-up of a classic soccer ball on green grass. WHITE hexagonal panels and BLACK pentagonal panels in the traditional pattern. Crisp black and white. Sharp focus sports photography.",
    "A person's foot about to kick a BLACK AND WHITE soccer ball. Classic pattern clearly visible. Action sports photography on grass.",
    "Close-up of a soccer ball in a goal net. Classic BLACK AND WHITE pattern: black pentagons, white hexagons. Stadium background. Professional photography.",
  ]},
  "basketball":{"name":"篮球","prompts":[
    "A wide shot of a basketball game in a gymnasium. Players on court, an ORANGE basketball mid-air, small in the frame. The ball is standard ORANGE with BLACK seam lines. Indoor arena sports photography.",
    "An outdoor basketball court in a park. An ORANGE basketball sitting on the concrete near the hoop. The ball is small in the wide scene. Urban outdoor sports photography.",
    "Close-up of a standard ORANGE basketball on a hardwood court. BLACK seam lines creating the 8-panel pattern. Pebbled texture. Sharp focus sports photography.",
    "Close-up texture of a basketball surface. ORANGE pebbled leather with a BLACK channel line. Warm saturated orange, deep black groove. Studio macro photography.",
    "An ORANGE basketball held in one hand. Bright ORANGE with BLACK channel lines. Pebbled texture visible. Clean gym background. Sports photography.",
  ]},
  "tennis_ball":{"name":"网球","prompts":[
    "A wide shot of a tennis match on a clay court. A player serving, a small FLUORESCENT YELLOW-GREEN ball visible in the air. The ball is tiny in the frame against the court and crowd. Sports event photography.",
    "A tennis court overview. Two YELLOW-GREEN tennis balls on the baseline, small in the wide frame. Red clay surface, net visible. Sports facility photography.",
    "Close-up of a FLUORESCENT YELLOW-GREEN tennis ball on a hard court. Fuzzy felt surface, white curved seam line. Vivid optic yellow-green. Sharp sports photography.",
    "A tennis ball held before serving. FLUORESCENT YELLOW-GREEN fuzzy felt, ~6.5cm. Bright optic yellow-green clearly visible. Close-up action sports photo.",
    "Two YELLOW-GREEN tennis balls next to a racket on grass. Regulation fluorescent color, fuzzy felt. Close-up on green grass. Wimbledon-style photography.",
  ]},
  "piano_keyboard":{"name":"钢琴键盘","prompts":[
    "A wide shot of a concert hall with a grand piano on stage. The keyboard is visible in the distance showing WHITE natural keys and BLACK sharp keys in groups of 2 and 3. The piano is a medium element in the grand hall. Concert photography.",
    "A living room scene with an upright piano against the wall. The keyboard shows WHITE keys in front, BLACK keys raised behind in 2-and-3 grouping. The piano is part of the room setting. Interior photography.",
    "Close-up of a piano keyboard: wide WHITE natural keys, narrower BLACK sharp/flat keys raised behind in groups of TWO and THREE. Multiple octaves. Sharp focus music photography.",
    "Top-down close-up of piano keys. WHITE keys longer and wider. BLACK keys in repeating groups: 2 black, then 3 black. Grand piano polished surface. Music photography.",
    "A pianist's hands on standard keyboard. WHITE keys lower, BLACK keys raised in 2-then-3 grouping. Clean polished keys. Performance photography.",
  ]},
  "poker_cards":{"name":"扑克花色","prompts":[
    "A wide shot of a poker table with multiple players. Playing cards scattered on the green felt, some showing RED hearts/diamonds, others BLACK spades/clubs. The cards are small in the casino scene. Casino photography.",
    "A card game in progress on a dining table. Several cards face-up showing RED heart symbols and BLACK spade symbols. Cards are medium-small in the domestic scene. Casual game photography.",
    "Close-up of four cards fanned out: heart (RED ♥), diamond (RED ♦), spade (BLACK ♠), club (BLACK ♣). Hearts/diamonds=RED, spades/clubs=BLACK. White background. Card photography.",
    "Two aces side by side: Ace of Hearts with RED heart, Ace of Spades with BLACK spade. Clear RED and BLACK distinction. On green felt. Casino close-up.",
    "Four Aces in a row: Hearts (RED), Diamonds (RED), Spades (BLACK), Clubs (BLACK). Red ink and black ink distinct. Product card photography.",
  ]},
}

# ===========================================================
# 有字母型 5 × 5 = 25
# ===========================================================
FASHION_TEXT = {
  "miumiu":{"name":"Miu Miu","prompts":[
    "A woman walking down a city street carrying a pink handbag. The bag has 'MIU MIU' text on the front but it is small and the woman is the main subject. Urban street style photography, full body shot.",
    "A boutique store interior with handbags on shelves. One pink bag with 'MIU MIU' text visible among many items. The text is small in the wider store scene. Retail interior photography.",
    "Close-up of a pink leather handbag front. Text 'MIU MIU' in BLACK capital serif letters. M-I-U space M-I-U crisp and legible. Luxury product photography.",
    "Close-up of a white fabric clothing label: 'MIU MIU' in black capital serif. M,I,U space M,I,U. Sharp print on woven label. Fashion detail macro.",
    "A leather wallet with 'MIU MIU' stamped in gold. M-I-U space M-I-U serif typeface. Crisp gold on smooth leather. Accessories photography.",
  ]},
  "gucci_text":{"name":"GUCCI","prompts":[
    "A person on a city street wearing a belt with a visible buckle area. The belt has 'GUCCI' text embossed but is small in the full-body street style photo. Urban fashion photography.",
    "A luxury store window display. A shopping bag with 'GUCCI' in gold letters visible among other items. The text is small in the wider storefront. Retail photography.",
    "Close-up of a leather belt: text 'GUCCI' embossed. G-U-C-C-I classic serif on brown leather. Sharp detail product photography.",
    "Close-up of a clothing label: 'GUCCI' in black capitals. G-U-C-C-I serif typeface on white fabric. Fashion detail photography.",
    "A perfume bottle labeled 'GUCCI' in capital letters. G-U-C-C-I refined serif on glass. Studio product photography.",
  ]},
  "dior_text":{"name":"DIOR","prompts":[
    "A woman carrying a canvas tote on a Parisian street. The bag has 'DIOR' embroidered on it but the text is relatively small in the street scene. Full body fashion street photography.",
    "A department store beauty counter. A perfume bottle with 'DIOR' gold lettering visible among many products. The text is small in the retail scene. Store interior photography.",
    "Close-up of canvas tote: 'DIOR' embroidered in navy serif on beige. D-I-O-R detailed stitching. Luxury product photography.",
    "Close-up of perfume bottle: 'DIOR' in gold. D-I-O-R elegant serif on glass. Studio photography warm lighting.",
    "A lipstick tube: 'DIOR' in silver letters. D-I-O-R thin serif. Silver on dark tube. Beauty product photography.",
  ]},
  "prada":{"name":"PRADA","prompts":[
    "A person carrying a black nylon bag on a city sidewalk. The bag has a small metal inverted triangle with 'PRADA' text, tiny in the full-body street photo. Urban fashion photography.",
    "A store shelf with multiple bags. One black bag has a small metal triangle reading 'PRADA'. The triangle is small among the products. Retail interior photography.",
    "Close-up of a metal inverted triangle badge on black nylon bag. 'PRADA' in capitals, P-R-A-D-A engraved. 'MILANO' smaller below. Silver on black. Fashion detail macro.",
    "Close-up of sunglasses temple: 'PRADA' engraved. P-R-A-D-A small capitals on acetate frame. Product photography marble surface.",
    "A leather wallet: 'PRADA' stamped in gold. P-R-A-D-A clean font on black saffiano leather. Accessories photography.",
  ]},
  "balenciaga":{"name":"BALENCIAGA","prompts":[
    "A person in an oversized black hoodie walking on a street. 'BALENCIAGA' in white letters across the chest but readable only at distance. Full body street style photography.",
    "A sneaker store display shelf. A black shoe with 'BALENCIAGA' text on the side, small in the row of many shoes. Retail interior photography.",
    "Close-up of black hoodie chest: 'BALENCIAGA' in large WHITE capitals. B-A-L-E-N-C-I-A-G-A bold sans-serif stretched wide. Fashion product photography.",
    "Close-up of sneaker side: 'BALENCIAGA' in small capitals. B-A-L-E-N-C-I-A-G-A white sans-serif on black. Sneaker product photography.",
    "A leather bag: 'BALENCIAGA' stamped on front. B-A-L-E-N-C-I-A-G-A capitals. Black on white leather. Luxury product photography.",
  ]},
}

# ===========================================================
# 图形符号型 3 × 5 = 15
# 不提品牌名, logo 描述详细
# ===========================================================
FASHION_GRAPHIC = {
  "ralph_lauren":{"name":"Ralph Lauren 骑手","prompts":[
    # 远景
    "A man wearing a navy polo shirt walking in a park. On his left chest there is a very small embroidered figure of a horseback rider facing RIGHT, swinging a mallet. The logo is tiny, about 2cm. The man and park are the main subject. Full body outdoor photography.",
    "A retail display showing folded polo shirts in various colors. Each shirt has a tiny embroidered polo rider on the chest facing RIGHT with mallet. The logos are small in the wider shelf scene. Retail store photography.",
    # 近景
    "Close-up of a navy polo shirt chest. Small embroidered silhouette: polo player on horseback at full gallop facing RIGHT, mallet swung upward. ~3cm, dark thread, crisp stitching. Sharp macro focus. Studio lighting.",
    "Macro of embroidered polo rider on white cotton. Horseback rider faces RIGHT, arm up with mallet, horse legs stretched running. Navy thread, visible stitches. Shallow depth of field.",
    "A cotton cap with embroidered polo player on front: horseback rider galloping RIGHT, mallet up. Contrasting thread. Close-up merchandise photography.",
  ]},
  "acne_studios":{"name":"Acne Studios 笑脸","prompts":[
    "A person wearing a knit beanie walking on a winter street. The hat has a very small square face patch on the front — barely visible in the full body shot. The face has two dot eyes and a straight line mouth. Street fashion photography.",
    "A clothing store shelf with folded sweatshirts. One has a tiny square face logo on the chest — two dot eyes, line mouth, square outline. The logo is small among the products. Retail photography.",
    "Close-up of a knit beanie with a small SQUARE-shaped FACE patch, ~2cm. The face is a SQUARE with TWO DOT EYES and a STRAIGHT HORIZONTAL LINE MOUTH. Neutral/calm expression. Knitwear detail photography.",
    "Close-up of sweatshirt chest: embroidered SQUARE FACE logo. SQUARE shape, TWO DOT EYES, STRAIGHT LINE MOUTH. Minimalist geometric. Small precise embroidery. Fashion detail photography.",
    "Macro of fabric patch: SQUARE FACE. Perfect SQUARE outline containing TWO circular DOT EYES and one HORIZONTAL LINE MOUTH. Minimal geometric design on wool scarf. Detail photography.",
  ]},
  "burberry":{"name":"Burberry 骑士","prompts":[
    "A person in a trench coat walking on a London street. On a button of the coat there is a tiny embossed emblem of a knight on horseback carrying a flag. The emblem is very small in the full-body photo. Street fashion photography.",
    "A luxury store shelf with scarves. One scarf label has a small knight-on-horseback emblem with a flag, barely visible among the products. Retail interior photography.",
    "Close-up of a trench coat button showing an embossed emblem: an EQUESTRIAN KNIGHT on horseback facing LEFT, carrying a FLAG or lance held upward. The knight wears armor. The horse is in a walking pose. Detailed metal relief on a round button. Fashion detail macro photography.",
    "Close-up of a scarf label with a printed emblem: a knight in armor on horseback facing LEFT, holding a FLAG with a pennant. The horse has four legs visible. Fine line art print on fabric label. Textile detail photography.",
    "Close-up of a leather bag clasp with an engraved knight: armored rider on horseback facing LEFT, FLAG raised. Horse walking. Metal engraving on polished surface. Luxury accessories macro photography.",
  ]},
}

# ===========================================================
# Monogram 型 2 × 5 = 10
# ===========================================================
FASHION_MONOGRAM = {
  "chanel":{"name":"Chanel 双C","prompts":[
    "A woman carrying a black quilted bag on a city street. The bag's clasp shows two interlocking C's (back-to-back), but the clasp is very small in the full-body street photo. The bag and outfit are the main subject. Fashion street photography.",
    "A department store handbag display. A black quilted bag among many others. Its gold CC clasp (two C's back-to-back) is a tiny detail in the wide retail scene. Store interior photography.",
    "Close-up of gold clasp on black quilted bag: two interlocking C letters BACK-TO-BACK. One C opens LEFT, one opens RIGHT, overlapping in middle. Same size, symmetrical. Polished gold. Luxury macro photography.",
    "Close-up gold pendant: two C's interlocked BACK-TO-BACK (facing away). One left, one right, overlapping symmetrically. Same size. Gold metal. Jewelry photography dark background.",
    "Close-up compact case lid: two interlocking C's facing AWAY from each other (back-to-back). One left one right. Same size. Gold on black lacquer. Beauty product photography.",
  ]},
  "lv":{"name":"Louis Vuitton LV","prompts":[
    "A traveler pulling a brown monogram suitcase through an airport. The LV monogram pattern (L and V overlapping, V in front of L, with flower motifs) covers the bag but is seen from a distance. The person and airport are the main subject. Travel photography.",
    "A luxury store window with multiple bags. A brown monogram bag shows the LV pattern (V over L, flowers, stars) but is one item among many. The pattern is small in the retail scene. Store photography.",
    "Close-up of brown coated canvas: repeating monogram with L and V overlapping — V IN FRONT of L. Flower and star motifs. Brown/tan. Sharp detail luxury product photography.",
    "Close-up leather trunk corner: embossed L and V overlap, V IN FRONT of L. Classic serif. Gold on brown leather. Vintage luggage detail photography.",
    "Close-up wallet surface: printed LV monogram. L and V overlap with V IN FRONT of L. Flower and star motifs around. Brown on tan. Accessories photography.",
  ]},
}

# ===========================================================
# GENERATORS
# ===========================================================
def gen_gpt(prompt,path):
    from openai import OpenAI
    c=OpenAI(api_key=OPENAI_API_KEY)
    for m in["gpt-image-1.5","gpt-image-1"]:
        try:
            r=c.images.generate(model=m,prompt=prompt,size="1024x1024",quality="high",output_format="png",n=1)
            with open(path,"wb")as f:f.write(base64.b64decode(r.data[0].b64_json))
            return True,m
        except:continue
    return False,"all failed"

def gen_gemini(prompt,path):
    url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={GEMINI_API_KEY}"
    try:
        r=requests.post(url,json={"contents":[{"parts":[{"text":prompt}]}]},timeout=120)
        if r.status_code!=200:return False,f"HTTP {r.status_code}"
        for c in r.json().get("candidates",[]):
            for p in c.get("content",{}).get("parts",[]):
                if "inlineData" in p:
                    with open(path,"wb")as f:f.write(base64.b64decode(p["inlineData"]["data"]))
                    return True,"gemini-3.1"
        return False,"no image"
    except Exception as e:return False,str(e)[:120]

ALL={"industry":INDUSTRY,"fashion_text":FASHION_TEXT,"fashion_graphic":FASHION_GRAPHIC,"fashion_monogram":FASHION_MONOGRAM}

def run(model,cat_filter=None,dry_run=False):
    gen=gen_gpt if model=="gpt" else gen_gemini
    meta,ok,fail=[],0,0
    for cat,brands in ALL.items():
        if cat_filter and cat!=cat_filter:continue
        print(f"\n{'='*55}\n  {cat} ({len(brands)} 子类)\n{'='*55}")
        for bk,bi in brands.items():
            out=BASE_DIR/"original"/model/cat;out.mkdir(parents=True,exist_ok=True)
            print(f"\n  {bi['name']} ({bk})")
            for i,pr in enumerate(bi["prompts"]):
                fn=f"{bk}_orig_{i+1}_{model}.png";fp=out/fn
                if fp.exists():print(f"    [{i+1}/5] SKIP");continue
                if dry_run:print(f"    [{i+1}/5] DRY: {pr[:65]}...");continue
                print(f"    [{i+1}/5]",end=" ",flush=True)
                s,info=gen(pr,str(fp))
                if s:ok+=1;print(f"OK ({info})")
                else:fail+=1;print(f"FAIL ({info})")
                meta.append({"filename":fn,"category":cat,"brand":bi["name"],"brand_key":bk,"type":"original_ai","prompt":pr,"model":model,"success":s,"timestamp":datetime.now().isoformat()})
                time.sleep(2)
    if meta and not dry_run:
        md=BASE_DIR/"metadata";md.mkdir(parents=True,exist_ok=True);mf=md/f"originals_{model}.json"
        ex=json.load(open(mf))if mf.exists()else[];ex.extend(meta);json.dump(ex,open(mf,"w"),indent=2,ensure_ascii=False)
        print(f"\nMetadata → {mf}")
    print(f"\n{'='*55}\n  DONE: OK={ok} FAIL={fail} TOTAL={ok+fail}\n{'='*55}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",default="gpt",choices=["gpt","gemini"]);p.add_argument("--category",default=None,choices=["all","industry","fashion_text","fashion_graphic","fashion_monogram"]);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    run(a.model,None if a.category=="all"else a.category,a.dry_run)
