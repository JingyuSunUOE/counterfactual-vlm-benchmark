"""
VLM Bias Dataset — 反事实图生成 (最终版)
==========================================
18 子类, 远景+近景混合
模型: gpt-image-1.5 high / gemini-3.1-flash-image-preview

用法:
    python gen_cf_final.py --model gpt
    python gen_cf_final.py --model gpt --category industry
    python gen_cf_final.py --model gpt --category fashion_text
    python gen_cf_final.py --model gpt --category fashion_graphic
    python gen_cf_final.py --model gpt --category fashion_monogram
    python gen_cf_final.py --model gpt --brand prada
    python gen_cf_final.py --model gpt --brand chanel --mod_id 1
    python gen_cf_final.py --dry-run
"""

import os,json,time,base64,argparse,requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BASE_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "archive" / "vlm_bias_dataset"

load_dotenv(REPO_ROOT / ".env", override=True)
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY","")
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")

# ===========================================================
# 行业标准型 反事实
# ===========================================================
CF_INDUSTRY={
 "traffic_light":{"name":"交通灯","mods":[
  {"id":1,"type":"order_swap","desc":"绿上黄中红下","gt":{"order":"green-yellow-red"},"bias":{"order":"red-yellow-green"},"prompts":[
    "A busy intersection with a traffic light in the upper part of the frame. The signal shows GREEN on TOP, YELLOW in MIDDLE, RED on BOTTOM. Green at the top. Cars and buildings around. The traffic light is a medium element in the urban scene. Daytime street photography.",
    "Close-up of a traffic light: GREEN on TOP, YELLOW MIDDLE, RED BOTTOM. Top light is green, middle amber, bottom red. Modern LED. Clear sky behind. Sharp focus.",
    "A traffic light over a suburban road: GREEN on TOP, then YELLOW, then RED at bottom. Green in the highest position. Daytime.",
    "Close-up traffic signal mounted on pole: GREEN (top), YELLOW (center), RED (bottom). Non-standard arrangement. Clear background.",
  ]},
  {"id":2,"type":"order_swap","desc":"黄上红中绿下","gt":{"order":"yellow-red-green"},"bias":{"order":"red-yellow-green"},"prompts":[
    "A street intersection with traffic light showing YELLOW on TOP, RED MIDDLE, GREEN BOTTOM. Amber at the top. Urban background. The light is medium-sized in the scene.",
    "Close-up: YELLOW on TOP, RED MIDDLE, GREEN BOTTOM. Top light is amber. Modern signal. Clear background.",
    "Traffic light over road: YELLOW, RED, GREEN from top to bottom. Yellow highest. Daytime.",
  ]},
  {"id":3,"type":"order_swap","desc":"红上绿中黄下","gt":{"order":"red-green-yellow"},"bias":{"order":"red-yellow-green"},"prompts":[
    "Close-up traffic light: RED TOP, GREEN MIDDLE, YELLOW BOTTOM. Middle is green not yellow, bottom is yellow not green. Positions swapped.",
    "Traffic signal at intersection: RED top, GREEN center, YELLOW bottom. Green and yellow reversed from normal. Urban scene.",
  ]},
 ]},
 "road_sign":{"name":"路牌","mods":[
  {"id":1,"type":"color_change","desc":"STOP牌绿底","gt":{"color":"green"},"bias":{"color":"red"},"prompts":[
    "A residential intersection. An octagonal STOP sign on a pole, but the background color is GREEN instead of red. 'STOP' in white on GREEN. The sign is medium-sized in the street scene. Houses and trees around.",
    "Close-up octagonal STOP sign: 'STOP' in WHITE on GREEN background (NOT red). Standard shape, GREEN color. Pole-mounted, clear sky.",
    "Close-up: GREEN octagonal stop sign. White 'STOP' text, white border. Green background instead of red. Clean sign.",
    "A GREEN STOP sign at a junction. Octagonal, white text on green. Not the normal red. Roadside setting.",
  ]},
  {"id":2,"type":"color_change","desc":"STOP牌蓝底","gt":{"color":"blue"},"bias":{"color":"red"},"prompts":[
    "An intersection with a BLUE octagonal STOP sign. 'STOP' in white on BLUE. The sign is in a street scene with buildings.",
    "Close-up: octagonal 'STOP' sign, BLUE background, white letters. Standard shape but BLUE not red.",
    "BLUE octagonal stop sign at road junction. White 'STOP' on BLUE. Residential setting.",
  ]},
  {"id":3,"type":"color_change","desc":"STOP牌黄底","gt":{"color":"yellow"},"bias":{"color":"red"},"prompts":[
    "Close-up: octagonal STOP sign, YELLOW background with BLACK text 'STOP'. Standard shape, YELLOW color.",
    "YELLOW octagonal stop sign. Black 'STOP' on YELLOW. Unusual color. Roadside.",
  ]},
 ]},
 "pingpong_ball":{"name":"乒乓球","mods":[
  {"id":1,"type":"color_change","desc":"绿色","gt":{"color":"green"},"bias":{"color":"white/orange"},"prompts":[
    "A table tennis match scene. A GREEN ball (not white/orange) is visible on the green table. The green ball is small in the wider sports hall. Sports photography.",
    "Close-up: GREEN table tennis ball on green table. Bright GREEN, not white or orange. 40mm, smooth. Sports photography.",
    "Player holding GREEN ping pong ball between fingers. Vivid GREEN. Not standard color. Clean background.",
    "Three GREEN ping pong balls on table near net. All bright GREEN. Sports facility.",
  ]},
  {"id":2,"type":"color_change","desc":"蓝色","gt":{"color":"blue"},"bias":{"color":"white/orange"},"prompts":[
    "Close-up: BLUE table tennis ball on table. Bright BLUE not white/orange. Smooth sphere.",
    "BLUE ping pong ball next to paddle. Distinctly BLUE. Product photography.",
    "Player serving BLUE table tennis ball. Vivid BLUE. Sports action.",
  ]},
  {"id":3,"type":"color_change","desc":"紫色","gt":{"color":"purple"},"bias":{"color":"white/orange"},"prompts":[
    "Close-up: PURPLE table tennis ball on green table. Bright PURPLE/VIOLET. Not white/orange.",
    "PURPLE ball on blue table. Distinctly PURPLE. Product photography.",
    "Two PURPLE ping pong balls on table near paddle. Vivid PURPLE.",
  ]},
 ]},
 "football":{"name":"足球","mods":[
  {"id":1,"type":"color_change","desc":"全红","gt":{"color":"all red"},"bias":{"color":"black and white"},"prompts":[
    "A football match, players on grass. The ball is entirely RED — all panels red, no black patches. The red ball is visible on the field. Sports event photography.",
    "Close-up: soccer ball entirely RED. All panels RED. No black pentagons, no white hexagons. On grass. Sports photography.",
    "Person kicking a RED soccer ball. Entirely red, no black or white. Action sports.",
    "RED soccer ball in goal net. Completely red, no black and white pattern. Stadium.",
  ]},
  {"id":2,"type":"color_change","desc":"全蓝","gt":{"color":"all blue"},"bias":{"color":"black and white"},"prompts":[
    "Close-up: soccer ball entirely BLUE. All panels BLUE. No black or white. On grass.",
    "BLUE soccer ball on pitch. Solid BLUE. No black pentagons. Sports photography.",
    "Completely BLUE soccer ball. Every panel blue. No traditional pattern. On grass.",
  ]},
 ]},
 "basketball":{"name":"篮球","mods":[
  {"id":1,"type":"color_change","desc":"蓝色","gt":{"color":"blue"},"bias":{"color":"orange"},"prompts":[
    "Basketball game in gym. A BLUE basketball (not orange) mid-air, small in the action scene. Blue with black seams. Sports event photography.",
    "Close-up: BLUE basketball on hardwood. Bright BLUE not orange, black seam lines. Pebbled texture.",
    "Player holding BLUE basketball. BLUE with black grooves. Not orange. Gym background.",
    "Close-up BLUE basketball surface. Pebbled, black channel lines. BLUE not orange. Macro.",
  ]},
  {"id":2,"type":"color_change","desc":"绿色","gt":{"color":"green"},"bias":{"color":"orange"},"prompts":[
    "Close-up: GREEN basketball on court. Bright GREEN not orange, black channel lines. Pebbled.",
    "Player dribbling GREEN basketball. GREEN with black grooves. Not orange. Gym.",
    "GREEN basketball texture close-up. Pebbled, black seams. GREEN not orange.",
  ]},
 ]},
 "tennis_ball":{"name":"网球","mods":[
  {"id":1,"type":"color_change","desc":"红色","gt":{"color":"red"},"bias":{"color":"yellow-green"},"prompts":[
    "Tennis match on court. A RED ball (not yellow-green) visible in play. The red ball is small in the match scene. Sports event photography.",
    "Close-up: RED tennis ball on court. Bright RED fuzzy felt, white seam. Not yellow-green.",
    "Player holding RED tennis ball. RED felt, not fluorescent yellow. Close-up.",
    "Two RED tennis balls next to racket on grass. Bright RED felt. Not yellow-green.",
  ]},
  {"id":2,"type":"color_change","desc":"蓝色","gt":{"color":"blue"},"bias":{"color":"yellow-green"},"prompts":[
    "Close-up: BLUE tennis ball on court. Bright BLUE fuzzy felt. Not yellow-green.",
    "Player tossing BLUE tennis ball. BLUE felt. Not yellow. Sports.",
    "BLUE tennis ball on grass. Bright BLUE fuzzy felt. Not yellow-green.",
  ]},
 ]},
 "piano_keyboard":{"name":"钢琴键盘","mods":[
  {"id":1,"type":"color_invert","desc":"黑白反转","gt":{"color":"inverted"},"bias":{"color":"standard"},"prompts":[
    "A concert hall with a piano on stage. The keyboard has INVERTED colors — BLACK natural keys, WHITE sharp keys. Visible from audience distance. The reversed keyboard is part of the stage scene.",
    "Close-up inverted piano: natural keys BLACK, sharp/flat keys WHITE. Wide front keys all BLACK, narrow raised keys WHITE. Reversed. Music photography.",
    "Top-down inverted piano: normally-white keys now BLACK, normally-black now WHITE. Groups of 2 and 3 white raised keys on black.",
    "Pianist's hands on inverted keyboard: BLACK naturals front, WHITE sharps behind in 2-and-3 groups. Concert.",
  ]},
  {"id":2,"type":"pattern_change","desc":"黑键4个一组","gt":{"grouping":"4"},"bias":{"grouping":"2 and 3"},"prompts":[
    "Close-up piano: BLACK keys in groups of FOUR not 2-and-3. Every group has FOUR black keys together. Non-standard.",
    "Piano keyboard: black sharp keys in groups of FOUR. Four black, white, four black, repeating. Unusual pattern.",
  ]},
 ]},
 "poker_cards":{"name":"扑克花色","mods":[
  {"id":1,"type":"color_swap","desc":"黑心红桃","gt":{"heart":"black","spade":"red"},"bias":{"heart":"red","spade":"black"},"prompts":[
    "A poker table scene. Some cards show BLACK heart symbols (not red) and RED spade symbols (not black). Colors swapped. Cards medium-small in casino scene.",
    "Close-up: heart (♥) BLACK, spade (♠) RED. Colors swapped. Black hearts, red spades. Card photography.",
    "Two aces: Hearts with BLACK heart, Spades with RED spade. Colors reversed. Green felt.",
    "Cards with SWAPPED colors: BLACK hearts and RED spades. Opposite of standard.",
  ]},
  {"id":2,"type":"color_swap","desc":"黑方红梅","gt":{"diamond":"black","club":"red"},"bias":{"diamond":"red","club":"black"},"prompts":[
    "Close-up: diamond (♦) BLACK, club (♣) RED. Colors swapped. Black diamonds, red clubs.",
    "Ace of Diamonds BLACK diamond, Ace of Clubs RED club. Reversed. Green felt.",
    "Cards: BLACK diamonds and RED clubs. Opposite of standard. Photography.",
  ]},
 ]},
}

# ===========================================================
# 有字母型 反事实
# ===========================================================
CF_FASHION_TEXT={
 "miumiu":{"name":"Miu Miu","mods":[
  {"id":1,"type":"typo","desc":"MIU MLU","gt":{"text":"MIU MLU"},"bias":{"text":"MIU MIU"},"prompts":[
    "A woman on a street with a pink bag. The bag text reads 'MIU MLU' (M-I-U M-L-U), small in the scene. L where second I should be. Street fashion photography.",
    "Close-up pink handbag: 'MIU MLU' in black serif capitals. M-I-U space M-L-U. Second word MLU not MIU. Product photography.",
    "Clothing label: 'MIU MLU'. M-I-U, M-L-U. L instead of I. White tag. Fashion detail.",
    "Wallet stamped gold: 'MIU MLU'. M-I-U M-L-U. MLU not MIU. Accessories photography.",
  ]},
  {"id":2,"type":"typo","desc":"MIU NIU","gt":{"text":"MIU NIU"},"bias":{"text":"MIU MIU"},"prompts":[
    "Close-up handbag: 'MIU NIU'. M-I-U space N-I-U. Second word starts N not M. Serif on pink leather.",
    "Label: 'MIU NIU'. M-I-U, N-I-U. N different from M. White tag. Fashion photography.",
    "Store sign: 'MIU NIU'. M-I-U N-I-U. Second word begins N. Retail.",
  ]},
  {"id":3,"type":"typo","desc":"MUU MUU","gt":{"text":"MUU MUU"},"bias":{"text":"MIU MIU"},"prompts":[
    "Close-up handbag: 'MUU MUU'. M-U-U M-U-U. No I. Serif on pink leather.",
    "Label: 'MUU MUU'. M-U-U M-U-U. Double U no I. White tag.",
    "Sign: 'MUU MUU' large capitals. M-U-U each word. No I. Retail.",
  ]},
 ]},
 "gucci_text":{"name":"GUCCI","mods":[
  {"id":1,"type":"typo","desc":"GUCOI","gt":{"text":"GUCOI"},"bias":{"text":"GUCCI"},"prompts":[
    "A store front with 'GUCOI' in large letters, visible in the street scene. G-U-C-O-I. O where second C should be. Urban retail photography.",
    "Close-up belt embossed 'GUCOI'. G-U-C-O-I. Second C replaced by O. Serif on leather.",
    "Shopping bag gold: 'GUCOI'. G-U-C-O-I. O where second C. Retail photography.",
    "Perfume labeled 'GUCOI'. G-U-C-O-I. Fourth letter O not C. Studio.",
  ]},
  {"id":2,"type":"typo","desc":"GVCCI","gt":{"text":"GVCCI"},"bias":{"text":"GUCCI"},"prompts":[
    "Close-up belt: 'GVCCI'. G-V-C-C-I. U replaced by V. Serif on leather.",
    "Bag: 'GVCCI' gold. G-V-C-C-I. V not U. Retail photography.",
    "Store entrance: 'GVCCI' metal serif. G,V,C,C,I. V replaces U.",
  ]},
 ]},
 "dior_text":{"name":"DIOR","mods":[
  {"id":1,"type":"typo","desc":"DIDR","gt":{"text":"DIDR"},"bias":{"text":"DIOR"},"prompts":[
    "A street scene with a store showing 'DIDR' signage. D-I-D-R. O replaced by D. The sign is part of the urban backdrop.",
    "Close-up tote: 'DIDR' embroidered. D-I-D-R navy serif on beige. O→D. Product photography.",
    "Perfume: 'DIDR' gold. D-I-D-R. Third letter D not O. Studio.",
    "Lipstick: 'DIDR' silver. D-I-D-R. O replaced by D. Beauty photography.",
  ]},
  {"id":2,"type":"typo","desc":"OIOR","gt":{"text":"OIOR"},"bias":{"text":"DIOR"},"prompts":[
    "Close-up tote: 'OIOR'. O-I-O-R. D replaced by O. Canvas. Product photography.",
    "Perfume: 'OIOR'. O-I-O-R. First letter O not D. Studio.",
    "Sign: 'OIOR' capitals. O,I,O,R. Both D positions are O. Retail.",
  ]},
 ]},
 "prada":{"name":"PRADA","mods":[
  {"id":1,"type":"typo","desc":"PRODA","gt":{"text":"PRODA"},"bias":{"text":"PRADA"},"prompts":[
    "A person carrying a bag with a small triangle reading 'PRODA'. P-R-O-D-A. O instead of A in third position. The triangle is small in the street scene.",
    "Close-up triangle badge: 'PRODA'. P-R-O-D-A. O where A should be. Silver on black nylon.",
    "Sunglasses: 'PRODA' on temple. P-R-O-D-A. O instead of A. Product photography.",
    "Wallet: 'PRODA' gold stamp. P-R-O-D-A. Third letter O. Leather accessories.",
  ]},
  {"id":2,"type":"typo","desc":"PRABA","gt":{"text":"PRABA"},"bias":{"text":"PRADA"},"prompts":[
    "Close-up triangle: 'PRABA'. P-R-A-B-A. D replaced by B. Silver on nylon.",
    "Sunglasses: 'PRABA'. P-R-A-B-A. B where D should be. Product.",
    "Wallet: 'PRABA' gold. P-R-A-B-A. B instead of D. Accessories.",
  ]},
  {"id":3,"type":"typo","desc":"PRADO","gt":{"text":"PRADO"},"bias":{"text":"PRADA"},"prompts":[
    "Close-up triangle: 'PRADO'. P-R-A-D-O. Last letter O not A. Metal on nylon.",
    "Store sign: 'PRADO' large capitals. P,R,A,D,O. Final O replaces A. Retail.",
    "Bag: 'PRADO'. P-R-A-D-O. O at the end. Leather accessories.",
  ]},
 ]},
 "balenciaga":{"name":"BALENCIAGA","mods":[
  {"id":1,"type":"typo","desc":"BALEMCIAGA","gt":{"text":"BALEMCIAGA"},"bias":{"text":"BALENCIAGA"},"prompts":[
    "A person in hoodie on street. Chest text 'BALEMCIAGA' (M instead of N), small in full body shot. Street fashion.",
    "Close-up hoodie: 'BALEMCIAGA' white capitals. B-A-L-E-M-C-I-A-G-A. M replaces N. Bold sans-serif.",
    "Sneaker: 'BALEMCIAGA' on side. B-A-L-E-M-C-I-A-G-A. M not N. White on black.",
    "Bag: 'BALEMCIAGA' stamped. B-A-L-E-M-C-I-A-G-A. N is M. Black on white leather.",
  ]},
  {"id":2,"type":"typo","desc":"BALENCIGA","gt":{"text":"BALENCIGA"},"bias":{"text":"BALENCIAGA"},"prompts":[
    "Close-up hoodie: 'BALENCIGA' white capitals. B-A-L-E-N-C-I-G-A. Missing second A, only 9 letters. Sans-serif.",
    "Sneaker: 'BALENCIGA'. B-A-L-E-N-C-I-G-A. One letter missing. White on black.",
    "Label: 'BALENCIGA'. B-A-L-E-N-C-I-G-A. Missing A after I. Sans-serif tag.",
  ]},
 ]},
}

# ===========================================================
# 图形符号型 反事实 (不提品牌名)
# ===========================================================
CF_FASHION_GRAPHIC={
 "ralph_lauren":{"name":"polo骑手","mods":[
  {"id":1,"type":"direction_flip","desc":"骑手朝LEFT","gt":{"direction":"left"},"bias":{"direction":"right"},"prompts":[
    "A man in a polo shirt walking in a park. The tiny chest logo shows a horseback rider facing LEFT with mallet. The logo is small in the full body shot. Outdoor photography.",
    "Retail shelf with polo shirts. Each has a small rider logo facing LEFT. The logos are tiny in the store scene.",
    "Close-up navy polo chest: polo player on horseback facing LEFT, mallet up. Horse galloping left. Dark thread ~3cm. Sharp macro.",
    "Macro embroidery on white: rider faces LEFT, arm with mallet, horse galloping left. Navy thread. Shallow depth of field.",
    "Cap: polo player facing LEFT on horseback with mallet. Oriented toward left. Close-up merchandise.",
  ]},
  {"id":2,"type":"element_removal","desc":"骑手无球杆","gt":{"has_mallet":"no"},"bias":{"has_mallet":"yes"},"prompts":[
    "Close-up polo logo: rider on galloping horse facing right, NO mallet. Arm raised but EMPTY. No stick. Dark thread on navy.",
    "Macro embroidery: polo player facing right, arm up holding NOTHING. No mallet. Empty raised hand. Detailed stitching.",
    "Cap emblem: horseback rider right, NO mallet. Arms but no equipment. Horse gallops normally.",
  ]},
  {"id":3,"type":"element_swap","desc":"骑驴","gt":{"animal":"donkey"},"bias":{"animal":"horse"},"prompts":[
    "Close-up polo logo: rider on DONKEY (not horse) facing right, mallet swung. Animal has LONG EARS like donkey. Dark thread on navy.",
    "Macro: polo player on DONKEY right with mallet. Distinctly LONG FLOPPY EARS. Stockier body. Stitching.",
    "Cap: rider on DONKEY with long ears, right, mallet. Clearly donkey not horse. Prominent ears.",
  ]},
 ]},
 "burberry":{"name":"Burberry 骑士","mods":[
  {"id":1,"type":"direction_flip","desc":"骑士朝RIGHT(非左)","gt":{"direction":"right"},"bias":{"direction":"left"},"prompts":[
    "A person in trench coat on London street. Coat button has tiny knight emblem facing RIGHT (not left). The emblem is very small in the full body shot.",
    "Close-up coat button: EQUESTRIAN KNIGHT on horseback facing RIGHT. Carrying FLAG raised. Armor visible. Horse walking right. Metal relief on button.",
    "Close-up scarf label: knight on horseback facing RIGHT, FLAG held up. Horse oriented to the right. Fine line art on fabric.",
    "Close-up leather bag: engraved knight facing RIGHT, FLAG raised, horse walking right. Metal on polished surface.",
  ]},
  {"id":2,"type":"element_removal","desc":"骑士无旗帜","gt":{"has_flag":"no"},"bias":{"has_flag":"yes"},"prompts":[
    "Close-up button: knight on horseback facing left, but NO FLAG. No lance, no pennant. Just an armored rider on a horse. Empty hands. Metal relief.",
    "Scarf label: knight on horseback facing left, NO FLAG or lance. Just riding. No weapon or banner. Line art on fabric.",
    "Bag: engraved knight facing left, horse walking, but NO FLAG. Knight's hands empty. No pennant. Metal engraving.",
  ]},
  {"id":3,"type":"element_removal","desc":"只有马没有骑士","gt":{"has_rider":"no"},"bias":{"has_rider":"yes"},"prompts":[
    "Close-up button: just a HORSE walking with NO RIDER. No knight, no person on top. Empty saddle or bare back. Just the horse alone. Metal relief on button.",
    "Scarf label: a HORSE walking to the left with NO RIDER on its back. No knight, no armor, no person. Just the horse by itself. Line art.",
    "Bag: engraved HORSE without any rider. No knight sitting on it. Just a riderless horse. Metal engraving.",
  ]},
 ]},
 "loewe":{"name":"Loewe Anagram","mods":[
  {"id":1,"type":"count_change","desc":"三个L(非四个)","gt":{"count":"3"},"bias":{"count":"4"},"prompts":[
    "Close-up leather bag: embossed logo with THREE L letters in a cross pattern (not four). One arm missing. Geometric but asymmetric. Tan leather. Product macro.",
    "Wallet: gold-stamped logo, THREE L's crossing at center. Only three arms, one direction empty. Gold on tan. Accessories.",
    "Fabric label: THREE L letters in a diamond pattern, one arm absent. Not the full four. Fashion detail.",
  ]},
  {"id":2,"type":"rotation","desc":"旋转45度","gt":{"rotation":"45°"},"bias":{"rotation":"0°"},"prompts":[
    "Close-up leather bag: embossed Anagram ROTATED 45 DEGREES. Four L's tilted so the diamond becomes a square orientation. Rotated from standard position. Tan leather macro.",
    "Wallet: gold Anagram ROTATED 45°. The cross/diamond pattern turned so it sits on a flat edge instead of a point. Gold on leather.",
    "Label: Anagram turned 45 degrees from normal orientation. Four L's rotated. Fabric detail.",
  ]},
  {"id":3,"type":"separation","desc":"四L分开不重叠","gt":{"arrangement":"separated"},"bias":{"arrangement":"overlapping"},"prompts":[
    "Close-up leather: four L letters SEPARATED, NOT overlapping. Each L stands alone with gaps between them, arranged in a cross shape but not touching. Tan leather.",
    "Wallet: four L's NOT overlapping. Spaced apart in diamond layout with visible gaps. Gold on leather.",
    "Label: four separate L letters, NOT interlocking. Clear space between each. Not the usual overlapping pattern.",
  ]},
 ]},
}

# ===========================================================
# Monogram 型 反事实
# ===========================================================
CF_FASHION_MONOGRAM={
 "chanel":{"name":"双C","mods":[
  {"id":1,"type":"direction_flip","desc":"双C面对面","gt":{"orientation":"face-to-face"},"bias":{"orientation":"back-to-back"},"prompts":[
    "A woman with a black quilted bag on a street. The clasp shows two C's facing INWARD (face-to-face, not back-to-back). The clasp is small in the full body scene.",
    "Store display: bag with CC clasp, two C's facing TOWARDS each other (openings inward). Small detail among products.",
    "Close-up gold clasp: two C's FACE-TO-FACE. Both openings INWARD toward center. NOT back-to-back. Same size. Polished gold. Macro.",
    "Close-up pendant: two C's FACE-TO-FACE, openings inward. Curves toward center. Same size. Gold. Jewelry photography.",
    "Close-up compact lid: two C's openings INWARD (face-to-face). Not back-to-back. Same size. Gold on black.",
  ]},
  {"id":2,"type":"rotation","desc":"双C旋转90度","gt":{"rotation":"90°"},"bias":{"rotation":"0°"},"prompts":[
    "Close-up clasp: two C's ROTATED 90 DEGREES. Openings UP and DOWN not left/right. Turned sideways. Gold. Macro.",
    "Pendant: double-C ROTATED 90°. C's open vertically (up and down). Turned on side. Gold. Jewelry.",
    "Compact lid: two C's rotated 90° — openings UP and DOWN. Horizontal/sideways. Gold on black.",
  ]},
  {"id":3,"type":"asymmetry","desc":"一大一小C","gt":{"symmetry":"asymmetric"},"bias":{"symmetry":"symmetric"},"prompts":[
    "Close-up clasp: two C's, one MUCH LARGER than other. ASYMMETRIC. Big C + small C. NOT same size. Gold on black. Macro.",
    "Pendant: double-C ASYMMETRIC. One ~TWICE the size of other. NOT equal. Gold. Jewelry.",
    "Compact lid: two C's DIFFERENT SIZES. One large, one small. Unequal. Gold on black.",
  ]},
 ]},
 "lv":{"name":"LV","mods":[
  {"id":1,"type":"layer_swap","desc":"L在V前面","gt":{"layering":"L in front"},"bias":{"layering":"V in front"},"prompts":[
    "A traveler with monogram bag at airport. The LV pattern shows L IN FRONT OF V (reversed). Small in the travel scene.",
    "Close-up canvas: L and V overlap but L IN FRONT OF V. L covers V. Reversed layering. Flower motifs. Brown/tan.",
    "Leather trunk: L IN FRONT overlapping V. V behind L. Reversed depth. Gold on brown.",
    "Wallet: LV monogram, L IN FRONT of V. Reversed layers. Brown floral on tan.",
  ]},
  {"id":2,"type":"mirror","desc":"镜像VL","gt":{"order":"VL"},"bias":{"order":"LV"},"prompts":[
    "Close-up canvas: letters arranged V-L (V left, L right). Reversed from L-V. V first. Flower motifs. Brown/tan.",
    "Leather: VL monogram — V left, L right. Reversed order. Gold on brown.",
    "Wallet: VL pattern (V first, L second). Reversed. Brown on tan.",
  ]},
  {"id":3,"type":"separation","desc":"LV分开不重叠","gt":{"arrangement":"separated"},"bias":{"arrangement":"overlapping"},"prompts":[
    "Close-up canvas: L and V SIDE BY SIDE with GAP. NOT overlapping. Separate letters, space between. Flowers around. Brown/tan.",
    "Leather: L and V SEPARATED. Not overlapping. Clear gap. Stand apart. Gold on leather.",
    "Wallet: L and V NOT overlapping. Next to each other, visible space. Separated. Brown.",
  ]},
 ]},
 "gucci_gg":{"name":"双G","mods":[
  {"id":1,"type":"direction_change","desc":"两个G同向(都朝上)","gt":{"orientation":"same direction"},"bias":{"orientation":"opposite directions"},"prompts":[
    "A person with canvas bag on street. The GG pattern shows both G's facing the SAME direction (both upright), not one inverted. Small in scene. Street photography.",
    "Close-up belt buckle: two G letters both UPRIGHT, facing SAME direction. NOT one inverted. Both G openings face the same way. Gold metal. Accessories macro.",
    "Close-up canvas: GG pattern where BOTH G's face UPRIGHT (same direction). Neither is inverted. Both openings point right. Brown on tan. Product photography.",
    "Wallet: gold GG, both G's oriented the SAME WAY (upright). Not one up one down. Same direction. Gold on black leather.",
  ]},
  {"id":2,"type":"rotation","desc":"GG旋转90度","gt":{"rotation":"90°"},"bias":{"rotation":"0°"},"prompts":[
    "Close-up belt buckle: interlocking GG ROTATED 90 DEGREES. The double-G turned sideways. G openings point up and down instead of left/right. Gold. Macro.",
    "Canvas bag: GG pattern ROTATED 90°. Each interlocking pair turned on its side. Unusual orientation. Brown on tan.",
    "Wallet: gold GG ROTATED 90 degrees from standard. Sideways. Gold on black leather.",
  ]},
  {"id":3,"type":"separation","desc":"两G分开不交织","gt":{"arrangement":"separated"},"bias":{"arrangement":"interlocking"},"prompts":[
    "Close-up buckle: two G letters SIDE BY SIDE with a GAP. NOT interlocking. Separate G's, space between. Gold metal. Macro.",
    "Canvas: GG pattern but each pair shows two G's NOT overlapping. Separated with visible gap. Brown on tan.",
    "Wallet: two G's SEPARATED, not interlocking. Standing apart with space. Gold on black leather.",
  ]},
 ]},
 "ysl":{"name":"YSL","mods":[
  {"id":1,"type":"order_swap","desc":"顺序颠倒LSY","gt":{"order":"L-S-Y"},"bias":{"order":"Y-S-L"},"prompts":[
    "A woman with black bag on street. The gold monogram reads L on TOP, S in MIDDLE, Y at BOTTOM (reversed from Y-S-L). Small in scene. Street photography.",
    "Close-up bag clasp: gold monogram L on TOP, S crossing MIDDLE, Y at BOTTOM. Reversed order. NOT Y-S-L but L-S-Y. Polished gold on black. Luxury macro.",
    "Pendant: gold L-S-Y vertically. L top, S middle, Y bottom. Upside-down from standard. Gold metal. Jewelry.",
    "Wallet: embossed monogram L (top), S (middle), Y (bottom). Reversed from standard Y-S-L. Gold on grain leather.",
  ]},
  {"id":2,"type":"element_removal","desc":"只有YL没有S","gt":{"letters":"Y and L only"},"bias":{"letters":"Y, S, and L"},"prompts":[
    "Close-up clasp: gold monogram with only Y and L interlocking. NO S in the middle. Missing the S. Just two letters. Gold on black. Macro.",
    "Pendant: Y and L overlapping, NO S. Two letters only, S absent. Gold metal. Jewelry.",
    "Wallet: embossed Y and L interlocking but NO S. Middle letter missing. Gold on leather.",
  ]},
  {"id":3,"type":"separation","desc":"YSL三字母分开","gt":{"arrangement":"separated"},"bias":{"arrangement":"interlocking"},"prompts":[
    "Close-up clasp: Y, S, L as three SEPARATE letters with GAPS. NOT interlocking. Standing apart vertically. Gold on black. Macro.",
    "Pendant: Y S L three separate gold letters, NOT overlapping. Visible space between each. Gold metal.",
    "Wallet: Y, S, L SEPARATED. Not interlocking. Three distinct letters with gaps. Gold on leather.",
  ]},
 ]},
 "celine":{"name":"Celine Triomphe","mods":[
  {"id":1,"type":"direction_flip","desc":"双C背靠背(像Chanel)","gt":{"orientation":"back-to-back"},"bias":{"orientation":"face-to-face"},"prompts":[
    "A woman with tan bag on street. The clasp shows two C's BACK-TO-BACK (openings outward, like Chanel), not face-to-face. Small in scene. Street photography.",
    "Close-up bag clasp: two C letters BACK-TO-BACK, openings facing OUTWARD (away from each other). NOT face-to-face. Like Chanel orientation. Gold on tan leather. Luxury macro.",
    "Close-up canvas: pattern of C pairs BACK-TO-BACK. Openings face away from each other. NOT the face-to-face Triomphe. Brown on tan. Product photography.",
    "Wallet clasp: two C's BACK-TO-BACK. Facing AWAY from each other (outward). Gold on leather. Accessories.",
  ]},
  {"id":2,"type":"rotation","desc":"Triomphe旋转90度","gt":{"rotation":"90°"},"bias":{"rotation":"0°"},"prompts":[
    "Close-up clasp: Triomphe double-C ROTATED 90 DEGREES. The interlocking C's turned sideways. Openings point up and down. Gold on tan. Macro.",
    "Canvas pattern: Triomphe ROTATED 90°. Each C pair turned on its side. Unusual orientation. Brown on tan.",
    "Wallet: Triomphe ROTATED 90 degrees. Sideways C's. Gold on leather.",
  ]},
  {"id":3,"type":"asymmetry","desc":"一大一小C","gt":{"symmetry":"asymmetric"},"bias":{"symmetry":"symmetric"},"prompts":[
    "Close-up clasp: Triomphe with one C MUCH LARGER than the other. ASYMMETRIC. Not equal size. One big C, one small C. Gold on tan. Macro.",
    "Canvas: Triomphe pattern with UNEQUAL C's. One C noticeably bigger. Asymmetric pairs. Brown on tan.",
    "Wallet clasp: two C's of DIFFERENT SIZES. One large, one small. Not the symmetric Triomphe. Gold on leather.",
  ]},
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
                if "inlineData"in p:
                    with open(path,"wb")as f:f.write(base64.b64decode(p["inlineData"]["data"]))
                    return True,"gemini-3.1"
        return False,"no image"
    except Exception as e:return False,str(e)[:120]

ALL_CF={"industry":CF_INDUSTRY,"fashion_text":CF_FASHION_TEXT,"fashion_graphic":CF_FASHION_GRAPHIC,"fashion_monogram":CF_FASHION_MONOGRAM}

def run(model,cat_filter=None,brand_filter=None,mod_filter=None,dry_run=False):
    gen=gen_gpt if model=="gpt"else gen_gemini
    meta,ok,fail=[],0,0
    for cat,brands in ALL_CF.items():
        if cat_filter and cat!=cat_filter:continue
        print(f"\n{'='*55}\n  {cat}\n{'='*55}")
        for bk,bi in brands.items():
            if brand_filter and bk!=brand_filter:continue
            out=BASE_DIR/"counterfactual"/model/cat;out.mkdir(parents=True,exist_ok=True)
            for mod in bi["mods"]:
                if mod_filter and mod["id"]!=mod_filter:continue
                print(f"\n  {bi['name']} | mod{mod['id']}: {mod['desc']}")
                for i,pr in enumerate(mod["prompts"]):
                    fn=f"{bk}_cf_mod{mod['id']}_{mod['type']}_{i+1}_{model}.png";fp=out/fn
                    if fp.exists():print(f"    [{i+1}/{len(mod['prompts'])}] SKIP");continue
                    if dry_run:print(f"    [{i+1}/{len(mod['prompts'])}] DRY: {pr[:65]}...");continue
                    print(f"    [{i+1}/{len(mod['prompts'])}]",end=" ",flush=True)
                    s,info=gen(pr,str(fp))
                    if s:ok+=1;print(f"OK ({info})")
                    else:fail+=1;print(f"FAIL ({info})")
                    meta.append({"filename":fn,"category":cat,"brand":bi["name"],"brand_key":bk,"type":"counterfactual","mod_id":mod["id"],"mod_type":mod["type"],"mod_desc":mod["desc"],"ground_truth":mod["gt"],"bias_answer":mod["bias"],"prompt":pr,"model":model,"success":s,"timestamp":datetime.now().isoformat()})
                    time.sleep(2)
    if meta and not dry_run:
        md=BASE_DIR/"metadata";md.mkdir(parents=True,exist_ok=True);mf=md/f"cf_{model}.json"
        ex=json.load(open(mf))if mf.exists()else[];ex.extend(meta);json.dump(ex,open(mf,"w"),indent=2,ensure_ascii=False)
        print(f"\nMetadata → {mf}")
    print(f"\n{'='*55}\n  DONE: OK={ok} FAIL={fail} TOTAL={ok+fail}\n{'='*55}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",default="gpt",choices=["gpt","gemini"]);p.add_argument("--category",default=None,choices=["all","industry","fashion_text","fashion_graphic","fashion_monogram"]);p.add_argument("--brand",default=None);p.add_argument("--mod_id",type=int,default=None);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    run(a.model,None if a.category=="all"else a.category,a.brand,a.mod_id,a.dry_run)
