"""
Question templates for the VLM Bias benchmark.
"""

from copy import deepcopy
import hashlib
import random


QUESTION_DESIGN_VERSION = "fashion_industry_questions_v3"
BOTH_RAW_PROMPT_POLICY = "neutral_pair_difference_v2"
BOTH_RAW_SCORING_POLICY = "neutral_pair_difference_v2"
BOTH_RAW_PAIR_DIFFERENCE_PREFIX = (
    "Two related images are provided. Compare the visible target attribute across the two images "
    "and answer using only what differs or stays the same."
)


PRIOR_PRIME_PREFIX = (
    "{prior_statement} "
    "Based only on the visible image, answer the following question carefully.\n\n"
    "{original_question}"
)


DISPLAY_NAME_MAP = {
    "traffic_light": "traffic light",
    "road_sign": "STOP road sign",
    "basketball": "basketball",
    "tennis_ball": "tennis ball",
    "piano_keyboard": "piano keyboard",
    "poker_cards": "deck of playing cards",
    "ralph_lauren": "Ralph Lauren Polo logo",
    "burberry": "Burberry equestrian knight logo",
    "loewe": "Loewe Anagram logo",
    "chanel": "Chanel interlocking C logo",
    "lv": "Louis Vuitton LV monogram",
    "gucci_gg": "Gucci double G logo",
    "ysl": "YSL monogram",
    "celine": "Celine Triomphe logo",
}


EVALUATION_TARGETS = {
    ("basketball", 1): "the dominant surface color of the basketball",
    ("basketball", 2): "the dominant surface color of the basketball",
    ("basketball", 3): "the dominant surface color of the basketball",
    ("tennis_ball", 1): "the color of the tennis ball",
    ("tennis_ball", 2): "the color of the tennis ball",
    ("tennis_ball", 3): "the color of the tennis ball",
    ("road_sign", 1): "the background color of the STOP sign",
    ("road_sign", 2): "the background color of the STOP sign",
    ("road_sign", 3): "the background color of the STOP sign",
    ("traffic_light", 1): "the top-to-bottom order of the three light colors",
    ("traffic_light", 2): "the top-to-bottom order of the three light colors",
    ("traffic_light", 3): "the top-to-bottom order of the three light colors",
    ("poker_cards", 1): "the color of the heart suit symbol",
    ("poker_cards", 2): "the color of the diamond suit symbol",
    ("poker_cards", 3): "which of the four suit symbols appear in red",
    ("piano_keyboard", 1): "the color of the wide natural keys",
    ("piano_keyboard", 2): "the number of black keys in each repeating group",
    ("piano_keyboard", 3): "the color of the raised sharp/flat keys",
    ("ralph_lauren", 1): "which horizontal direction the polo rider faces",
    ("ralph_lauren", 2): "whether the polo rider is holding a mallet",
    ("ralph_lauren", 3): "what animal the polo rider is sitting on",
    ("burberry", 1): "which horizontal direction the equestrian knight faces",
    ("burberry", 2): "whether the knight is carrying a flag or lance",
    ("burberry", 3): "whether a rider is visible on the horse",
    ("loewe", 1): "the number of L letters in the logo",
    ("loewe", 2): "whether the logo is in standard upright orientation or rotated",
    ("loewe", 3): "whether the four L letters are interlocking or separated",
    ("chanel", 1): "the orientation of the two C letters (face-to-face or back-to-back)",
    ("chanel", 2): "the direction the C openings point (horizontal or vertical)",
    ("chanel", 3): "whether the two C letters are the same or different sizes",
    ("lv", 1): "which of L or V letter appears layered in front",
    ("lv", 2): "the left-to-right letter order in the monogram",
    ("lv", 3): "whether L and V are overlapping or separated",
    ("gucci_gg", 1): "whether the two G letters face opposite or the same direction",
    ("gucci_gg", 2): "whether the logo is in standard upright orientation or rotated",
    ("gucci_gg", 3): "whether the two G letters are interlocking or separated",
    ("ysl", 1): "the top-to-bottom letter order in the monogram",
    ("ysl", 2): "the number of distinct letters present in the monogram",
    ("ysl", 3): "whether the letters are interlocking or separated",
    ("celine", 1): "the orientation of the two C letters (face-to-face or back-to-back)",
    ("celine", 2): "whether the logo is in standard upright orientation or rotated",
    ("celine", 3): "whether the two C letters are the same or different sizes",
}


DISTRACTOR_POOLS = {
    "basketball": ["red", "yellow", "white", "brown"],
    "tennis_ball": ["orange", "white", "brown", "black"],
    "road_sign": ["white", "brown", "pink", "orange"],
    "traffic_light": ["blue-red-green", "yellow-green-red", "red-green-yellow", "green-red-yellow"],
    "poker_cards": ["yellow", "blue", "green", "purple"],
    "piano_keyboard": ["blue", "red", "green", "yellow"],
    "ralph_lauren": ["backward", "downward", "missing", "sideways"],
    "burberry": ["backward", "downward", "missing", "sideways"],
    "loewe": ["5", "2", "6", "1"],
    "chanel": ["up-down", "diagonal", "overlapping", "separated"],
    "lv": ["vertical-stack", "diagonal", "mirror-image", "rotated"],
    "gucci_gg": ["stacked", "diagonal", "mirrored", "overlapping"],
    "ysl": ["S-Y-L", "L-Y-S", "Y-L-S", "L-S-Y"],
    "celine": ["up-down", "diagonal", "overlapping", "separated"],
}


def _open_question(text):
    return f"{text} Answer inside curly brackets with ≤5 words, e.g. {{...}}."


def _pair_open_question(text):
    return f"{text} Answer inside curly brackets in one short phrase, e.g. {{...}}."


def _yes_no_question(text):
    return f"{text} Answer inside curly brackets, e.g. {{Yes}} or {{No}}."


def _answer(correct, biased):
    return {"correct": correct, "biased": biased}


def _design(prior_statement, q1, q2, q3, q1_answer, q2_answer=None, mc_answer=None, mc_distractors=None):
    q2_answer = q2_answer or q1_answer
    mc_answer = mc_answer or q1_answer
    return {
        "prior_statement": prior_statement,
        "questions": {
            "Q1": {"prompt": _open_question(q1), **q1_answer},
            "Q2": {"prompt": _open_question(q2), **q2_answer},
            "Q3": {"prompt": _yes_no_question(q3), "correct": "no", "biased": "yes"},
            "Q_MC": {**mc_answer, "distractors": mc_distractors},
        },
    }


QUESTION_DESIGNS = {
    ("traffic_light", 1): _design(
        "Traffic lights are commonly expected to be ordered red, yellow, then green from top to bottom.",
        "What is the visible top-to-bottom order of the three light colors?",
        "Describe the visible color sequence in this signal from top to bottom.",
        "Does the visible signal show the standard red-yellow-green order?",
        _answer("green-yellow-red", "red-yellow-green"),
    ),
    ("traffic_light", 2): _design(
        "Traffic lights are commonly expected to be ordered red, yellow, then green from top to bottom.",
        "What is the visible top-to-bottom order of the three light colors?",
        "Describe the visible color sequence in this signal from top to bottom.",
        "Does the visible signal show the standard red-yellow-green order?",
        _answer("yellow-red-green", "red-yellow-green"),
    ),
    ("traffic_light", 3): _design(
        "Traffic lights are commonly expected to be ordered red, yellow, then green from top to bottom.",
        "What is the visible top-to-bottom order of the three light colors?",
        "Describe the visible color sequence in this signal from top to bottom.",
        "Does the visible signal show the standard red-yellow-green order?",
        _answer("red-green-yellow", "red-yellow-green"),
    ),
    ("road_sign", 1): _design(
        "STOP signs are commonly expected to have a red background.",
        "What is the visible background color of the STOP sign?",
        "Describe the visible background color of the octagonal sign.",
        "Does the visible STOP sign have the normally expected red background?",
        _answer("green", "red"),
    ),
    ("road_sign", 2): _design(
        "STOP signs are commonly expected to have a red background.",
        "What is the visible background color of the STOP sign?",
        "Describe the visible background color of the octagonal sign.",
        "Does the visible STOP sign have the normally expected red background?",
        _answer("blue", "red"),
    ),
    ("road_sign", 3): _design(
        "STOP signs are commonly expected to have a red background.",
        "What is the visible background color of the STOP sign?",
        "Describe the visible background color of the octagonal sign.",
        "Does the visible STOP sign have the normally expected red background?",
        _answer("yellow", "red"),
    ),
    ("basketball", 1): _design(
        "Basketballs are commonly expected to have an orange surface.",
        "What is the visible surface color of the basketball?",
        "Describe the visible overall color of the basketball.",
        "Does the basketball have the normally expected orange surface?",
        _answer("blue", "orange"),
    ),
    ("basketball", 2): _design(
        "Basketballs are commonly expected to have an orange surface.",
        "What is the visible surface color of the basketball?",
        "Describe the visible overall color of the basketball.",
        "Does the basketball have the normally expected orange surface?",
        _answer("green", "orange"),
    ),
    ("basketball", 3): _design(
        "Basketballs are commonly expected to have an orange surface.",
        "What is the visible surface color of the basketball?",
        "Describe the visible overall color of the basketball.",
        "Does the basketball have the normally expected orange surface?",
        _answer("purple", "orange"),
    ),
    ("tennis_ball", 1): _design(
        "Tennis balls are commonly expected to be yellow-green.",
        "What is the visible color of the tennis ball?",
        "Describe the visible color of the ball.",
        "Does the ball have the normally expected yellow-green color?",
        _answer("red", "yellow-green"),
    ),
    ("tennis_ball", 2): _design(
        "Tennis balls are commonly expected to be yellow-green.",
        "What is the visible color of the tennis ball?",
        "Describe the visible color of the ball.",
        "Does the ball have the normally expected yellow-green color?",
        _answer("blue", "yellow-green"),
    ),
    ("tennis_ball", 3): _design(
        "Tennis balls are commonly expected to be yellow-green.",
        "What is the visible color of the tennis ball?",
        "Describe the visible color of the ball.",
        "Does the ball have the normally expected yellow-green color?",
        _answer("pink", "yellow-green"),
    ),
    ("piano_keyboard", 1): _design(
        "Piano keyboards are commonly expected to have white wide natural keys and black raised sharp/flat keys.",
        "What color are the visible wide natural keys on this keyboard?",
        "Describe the visible colors of the wide natural keys and the raised keys.",
        "Does the keyboard show the standard white natural keys and black raised keys?",
        _answer("black", "white"),
        q2_answer=_answer("black natural keys and white raised keys", "white natural keys and black raised keys"),
        mc_answer=_answer("black", "white"),
    ),
    ("piano_keyboard", 2): _design(
        "Piano keyboards are commonly expected to have black keys arranged in alternating groups of two and three.",
        "How many raised black keys appear in each visible repeating group?",
        "Describe the visible grouping pattern of the raised keys.",
        "Does the keyboard show the standard alternating groups of two and three black keys?",
        _answer("4", "2 and 3"),
    ),
    ("piano_keyboard", 3): _design(
        "Piano keyboards are commonly expected to have black raised sharp/flat keys.",
        "What color are the visible raised sharp/flat keys on this keyboard?",
        "Describe the visible color of the narrow raised keys.",
        "Do the raised sharp/flat keys have the normally expected black color?",
        _answer("red", "black"),
    ),
    ("poker_cards", 1): _design(
        "Playing-card hearts are commonly expected to be red, while spades are commonly expected to be black.",
        "What color is the visible heart suit symbol on these cards?",
        "Describe the visible colors of the heart and spade suit symbols.",
        "Do the suit symbols follow the standard red hearts and black spades colors?",
        _answer("black", "red"),
        q2_answer=_answer("black hearts and red spades", "red hearts and black spades"),
    ),
    ("poker_cards", 2): _design(
        "Playing-card diamonds are commonly expected to be red, while clubs are commonly expected to be black.",
        "What color is the visible diamond suit symbol on these cards?",
        "Describe the visible colors of the diamond and club suit symbols.",
        "Do the suit symbols follow the standard red diamonds and black clubs colors?",
        _answer("black", "red"),
        q2_answer=_answer("black diamonds and red clubs", "red diamonds and black clubs"),
    ),
    ("poker_cards", 3): _design(
        "Playing-card suits are commonly expected to include red hearts and diamonds.",
        "How many red suit symbols are visible across these cards?",
        "Describe whether any suit symbols are visibly red.",
        "Is at least one red suit symbol visible?",
        _answer("0", "2"),
        q2_answer=_answer("no red suit symbols", "red hearts and diamonds"),
        mc_answer=_answer("none", "hearts and diamonds"),
        mc_distractors=["hearts only", "diamonds only", "all four suits"],
    ),
    ("ralph_lauren", 1): _design(
        "The Ralph Lauren Polo rider is commonly expected to face right.",
        "Which horizontal direction does the visible polo rider face?",
        "Describe the visible direction of the horseback rider.",
        "Does the visible polo rider face the normally expected right direction?",
        _answer("left", "right"),
    ),
    ("ralph_lauren", 2): _design(
        "The Ralph Lauren Polo rider is commonly expected to hold a mallet.",
        "Does the visible rider hold any mallet or polo stick?",
        "Describe whether anything is visible in the rider's raised hand.",
        "Does the visible rider have the normally expected mallet?",
        _answer("no", "yes"),
    ),
    ("ralph_lauren", 3): _design(
        "The Ralph Lauren Polo rider is commonly expected to sit on a horse.",
        "What animal is visible under the rider?",
        "Describe the visible animal in the emblem.",
        "Is the visible rider sitting on the normally expected horse?",
        _answer("donkey", "horse"),
    ),
    ("burberry", 1): _design(
        "The Burberry equestrian knight is commonly expected to face left.",
        "Which horizontal direction does the visible equestrian knight face?",
        "Describe the visible direction of the knight.",
        "Does the visible knight face the normally expected left direction?",
        _answer("right", "left"),
    ),
    ("burberry", 2): _design(
        "The Burberry equestrian knight is commonly expected to carry a flag or lance.",
        "Does the visible knight carry any flag or lance?",
        "Describe what, if anything, the visible knight is holding.",
        "Does the visible knight have the normally expected flag or lance?",
        _answer("no", "yes"),
    ),
    ("burberry", 3): _design(
        "The Burberry emblem is commonly expected to show a rider on the horse.",
        "Is any rider visibly present on the horse?",
        "Describe who or what is visibly on the horse.",
        "Is the normally expected knight visibly riding the horse?",
        _answer("no", "yes"),
    ),
    ("loewe", 1): _design(
        "The Loewe Anagram is commonly expected to have four L-shaped letters.",
        "How many L-shaped letters are visible in this logo?",
        "Count the visible L-shaped elements in the logo.",
        "Does the visible logo have the normally expected four L letters?",
        _answer("3", "4"),
    ),
    ("loewe", 2): _design(
        "The Loewe Anagram is commonly expected to appear in its standard upright orientation.",
        "Describe the visible orientation of this logo.",
        "Is the visible logo upright or tilted?",
        "Does the visible logo have the normally expected upright orientation?",
        _answer("rotated 45 degrees", "standard upright"),
        q2_answer=_answer("tilted", "upright"),
        mc_answer=_answer("45°", "0°"),
    ),
    ("loewe", 3): _design(
        "The Loewe Anagram is commonly expected to have interlocking L-shaped letters.",
        "Describe how the visible L-shaped letters are arranged relative to each other.",
        "Are the visible L-shaped letters touching each other or spaced apart?",
        "Are the visible L-shaped letters in the normally expected interlocking arrangement?",
        _answer("separated", "overlapping"),
    ),
    ("chanel", 1): _design(
        "The Chanel interlocking C logo is commonly expected to have two C letters back-to-back.",
        "Describe the visible orientation of the two C letters.",
        "Which way do the visible C openings face relative to each other?",
        "Do the visible C letters have the normally expected back-to-back orientation?",
        _answer("face-to-face", "back-to-back"),
    ),
    ("chanel", 2): _design(
        "The Chanel interlocking C logo is commonly expected to have horizontally oriented C openings.",
        "Which directions do the visible C openings point?",
        "Describe the visible orientation of the C openings.",
        "Do the visible C openings point in the normally expected horizontal directions?",
        _answer("up-down", "left-right"),
        mc_answer=_answer("90°", "0°"),
    ),
    ("chanel", 3): _design(
        "The Chanel interlocking C logo is commonly expected to have two C letters of the same size.",
        "Describe the visible relative sizes of the two C letters.",
        "Are the visible C letters equal in size or unequal?",
        "Do the visible C letters have the normally expected matching sizes?",
        _answer("different sizes", "same size"),
        mc_answer=_answer("asymmetric", "symmetric"),
    ),
    ("lv", 1): _design(
        "The Louis Vuitton LV monogram is commonly expected to show the V layered in front of the L.",
        "Which visible letter appears layered in front?",
        "Describe the visible layering of the L and V letters.",
        "Does the visible monogram show the normally expected V in front?",
        _answer("L in front", "V in front"),
    ),
    ("lv", 2): _design(
        "The Louis Vuitton monogram is commonly expected to read L then V from left to right.",
        "What is the visible left-to-right letter order in this monogram?",
        "Describe the visible order of the letters.",
        "Does the visible monogram show the normally expected L-V order?",
        _answer("VL", "LV"),
    ),
    ("lv", 3): _design(
        "The Louis Vuitton LV monogram is commonly expected to have overlapping letters.",
        "Describe whether the visible L and V letters overlap or are separated.",
        "How are the visible L and V letters arranged relative to each other?",
        "Do the visible L and V letters have the normally expected overlapping arrangement?",
        _answer("separated", "overlapping"),
    ),
    ("gucci_gg", 1): _design(
        "The Gucci double-G logo is commonly expected to have the two G letters facing opposite directions.",
        "Describe the visible orientation of the two G letters.",
        "Do the visible G letters face matching directions or opposing directions?",
        "Do the visible G letters have the normally expected opposite-facing orientation?",
        _answer("same direction", "opposite directions"),
    ),
    ("gucci_gg", 2): _design(
        "The Gucci double-G logo is commonly expected to appear in its standard upright orientation.",
        "Describe the visible orientation of this double-G logo.",
        "Is the visible logo upright or turned sideways?",
        "Does the visible double-G logo have the normally expected upright orientation?",
        _answer("rotated 90 degrees", "standard upright"),
        q2_answer=_answer("turned sideways", "upright"),
        mc_answer=_answer("90°", "0°"),
    ),
    ("gucci_gg", 3): _design(
        "The Gucci double-G logo is commonly expected to have interlocking G letters.",
        "Describe how the visible G letters are arranged relative to each other.",
        "Are the visible G letters touching each other or spaced apart?",
        "Do the visible G letters have the normally expected interlocking arrangement?",
        _answer("separated", "interlocking"),
    ),
    ("ysl", 1): _design(
        "The YSL monogram is commonly expected to read Y, S, then L from top to bottom.",
        "What is the visible top-to-bottom letter order in this monogram?",
        "Describe the visible vertical arrangement of the letters.",
        "Does the visible monogram have the normally expected Y-S-L order?",
        _answer("L-S-Y", "Y-S-L"),
    ),
    ("ysl", 2): _design(
        "The YSL monogram is commonly expected to contain the Y, S, and L letters.",
        "How many distinct letters are visible in this monogram?",
        "Which letters are visibly present in this monogram?",
        "Does the visible monogram contain the normally expected S letter?",
        _answer("2", "3"),
        q2_answer=_answer("Y and L only", "Y, S, and L"),
        mc_answer=_answer("Y and L only", "Y, S, and L"),
        mc_distractors=["Y only", "L only", "S and L only"],
    ),
    ("ysl", 3): _design(
        "The YSL monogram is commonly expected to have interlocking letters.",
        "Describe how the visible letters are arranged relative to each other.",
        "Are the visible letters touching each other or spaced apart?",
        "Do the visible letters have the normally expected interlocking arrangement?",
        _answer("separated", "interlocking"),
    ),
    ("celine", 1): _design(
        "The Celine Triomphe logo is commonly expected to have two C letters facing each other.",
        "Describe the visible orientation of the two C letters.",
        "Which way do the visible C openings face relative to each other?",
        "Do the visible C letters have the normally expected face-to-face orientation?",
        _answer("back-to-back", "face-to-face"),
    ),
    ("celine", 2): _design(
        "The Celine Triomphe logo is commonly expected to appear in its standard upright orientation.",
        "Describe the visible orientation of this Triomphe logo.",
        "Is the visible logo upright or turned sideways?",
        "Does the visible Triomphe logo have the normally expected upright orientation?",
        _answer("rotated 90 degrees", "standard upright"),
        q2_answer=_answer("turned sideways", "upright"),
        mc_answer=_answer("90°", "0°"),
    ),
    ("celine", 3): _design(
        "The Celine Triomphe logo is commonly expected to have two C letters of the same size.",
        "Describe the visible relative sizes of the two C letters.",
        "Are the visible C letters equal in size or unequal?",
        "Do the visible C letters have the normally expected matching sizes?",
        _answer("different sizes", "same size"),
        mc_answer=_answer("asymmetric", "symmetric"),
    ),
}

def _mc_rng(key, mod_id):
    seed_material = f"{key}_{mod_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    return random.Random(seed)


def _short_evaluation_target(key, mod_id):
    target = EVALUATION_TARGETS[(key, mod_id)]
    if target.startswith("the "):
        return target[4:]
    return target


def prior_statement_for(key, mod_id=None):
    if mod_id is not None and (key, mod_id) in QUESTION_DESIGNS:
        return QUESTION_DESIGNS[(key, mod_id)]["prior_statement"]
    display_name = DISPLAY_NAME_MAP[key]
    return f"{display_name} has a familiar canonical visual appearance."


def _format_question_with_prior(question, key, mod_id=None, prime=True):
    if not prime:
        return question
    return PRIOR_PRIME_PREFIX.format(
        prior_statement=prior_statement_for(key, mod_id),
        original_question=question,
    )


def _format_pair_question_with_prior(question, key, mod_id=None, prime=True):
    if not prime:
        return question
    return (
        f"{prior_statement_for(key, mod_id)} "
        "Based only on the visible images, answer the following comparison question carefully.\n\n"
        f"{question}"
    )


def is_both_raw_pair_difference(input_mode, tool_condition="raw"):
    return input_mode == "both" and tool_condition == "raw"


def both_raw_policy_fields(input_mode, tool_condition="raw"):
    if is_both_raw_pair_difference(input_mode, tool_condition):
        return {
            "both_raw_prompt_policy": BOTH_RAW_PROMPT_POLICY,
            "both_raw_scoring_policy": BOTH_RAW_SCORING_POLICY,
        }
    return {
        "both_raw_prompt_policy": None,
        "both_raw_scoring_policy": None,
    }


def _pair_difference_open_question(key, mod_id, question_type):
    target = _short_evaluation_target(key, mod_id)
    if question_type == "Q3":
        return _yes_no_question(f"Do the two related images show the same {target}?")
    if question_type == "Q2":
        return _pair_open_question(
            f"Name the visible values or states for {target} across the two related images. "
            "If they appear the same, answer {same}."
        )
    return _pair_open_question(
        f"Compare the two related images. Is there a visible difference in {target}? "
        "If yes, briefly describe the difference."
    )


def _pair_difference_mc_prompt(key, mod_id, prime=True):
    target = _short_evaluation_target(key, mod_id)
    options = {
        "A": "The two images show the same visible target attribute.",
        "B": "The two images show a visible difference in the target attribute.",
        "C": "The relevant target attribute is unclear in both images.",
        "D": "Only unrelated background or context differs.",
    }
    question = (
        f"Which option best describes {target} across the two related images?\n"
        f"A) {options['A']}\n"
        f"B) {options['B']}\n"
        f"C) {options['C']}\n"
        f"D) {options['D']}\n\n"
        "Answer with only the single capital letter A, B, C, or D."
    )
    return {
        "prompt": _format_pair_question_with_prior(question, key, mod_id=mod_id, prime=prime),
        "options": options,
        "correct_letter": "B",
        "biased_letter": "A",
    }


def build_mc_prompt(
    key,
    mod_id,
    correct_value,
    biased_value,
    input_mode,
    prime=True,
    distractor_pool=None,
):
    rng = _mc_rng(key, mod_id)
    distractor_pool = DISTRACTOR_POOLS[key] if distractor_pool is None else distractor_pool
    pool = [
        value
        for value in distractor_pool
        if value not in {correct_value, biased_value}
    ]

    option_values = [correct_value]
    if biased_value is not None and biased_value != correct_value:
        option_values.append(biased_value)

    needed = 4 - len(option_values)
    if len(pool) < needed:
        raise ValueError(f"Not enough distractors for {key} mod {mod_id}")

    distractors = rng.sample(pool, needed)
    option_values.extend(distractors)
    rng.shuffle(option_values)

    letters = ("A", "B", "C", "D")
    options = dict(zip(letters, option_values))
    correct_letter = next(letter for letter, value in options.items() if value == correct_value)

    biased_letter = None
    if input_mode != "orig_only" and biased_value is not None and biased_value != correct_value:
        biased_letter = next(letter for letter, value in options.items() if value == biased_value)

    question = (
        f"Looking at this image carefully, which of the following best describes "
        f"{_short_evaluation_target(key, mod_id)}?\n"
        f"A) {options['A']}\n"
        f"B) {options['B']}\n"
        f"C) {options['C']}\n"
        f"D) {options['D']}\n\n"
        "Answer with only the single capital letter A, B, C, or D."
    )
    if prime:
        question = _format_question_with_prior(question, key, mod_id=mod_id, prime=True)

    return {
        "prompt": question,
        "options": options,
        "correct_letter": correct_letter,
        "biased_letter": biased_letter,
    }


EVAL_INDUSTRY = {
    "traffic_light": {
        "sanity": {
            "Q1": _open_question("What is the order of colors on this traffic light from top to bottom?"),
            "Q2": _open_question("Describe the arrangement of the three lights on this traffic light from top to bottom."),
            "Q3": _yes_no_question("Is this traffic light arranged with red on top, yellow in middle, and green on bottom?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What is the order of colors on this traffic light from top to bottom?"),
                "Q2": _open_question("Describe the arrangement of the three lights on this traffic light from top to bottom."),
                "Q3": _yes_no_question("Is this traffic light arranged with red on top, yellow in middle, and green on bottom?"),
            },
            2: {
                "Q1": _open_question("What is the order of colors on this traffic light from top to bottom?"),
                "Q2": _open_question("Describe the arrangement of the three lights on this traffic light from top to bottom."),
                "Q3": _yes_no_question("Is this traffic light arranged with red on top, yellow in middle, and green on bottom?"),
            },
            3: {
                "Q1": _open_question("What is the order of colors on this traffic light from top to bottom?"),
                "Q2": _open_question("Describe the arrangement of the three lights on this traffic light from top to bottom."),
                "Q3": _yes_no_question("Is this traffic light arranged with red on top, yellow in middle, and green on bottom?"),
            },
        },
    },
    "road_sign": {
        "sanity": {
            "Q1": _open_question("What color is the background of this stop sign?"),
            "Q2": _open_question("What single word is written on this sign in capital letters?"),
            "Q3": _yes_no_question("Is this stop sign red?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What color is the background of this stop sign?"),
                "Q2": _open_question("What single word is written on this sign in capital letters?"),
                "Q3": _yes_no_question("Is this stop sign red?"),
            },
            2: {
                "Q1": _open_question("What color is the background of this stop sign?"),
                "Q2": _open_question("What single word is written on this sign in capital letters?"),
                "Q3": _yes_no_question("Is this stop sign red?"),
            },
            3: {
                "Q1": _open_question("What color is the background of this stop sign?"),
                "Q2": _open_question("What single word is written on this sign in capital letters?"),
                "Q3": _yes_no_question("Is this stop sign red?"),
            },
        },
    },
    "basketball": {
        "sanity": {
            "Q1": _open_question("What color is this basketball?"),
            "Q2": _open_question("Describe the color of this basketball."),
            "Q3": _yes_no_question("Is this an orange basketball?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What color is this basketball?"),
                "Q2": _open_question("Describe the overall color of this basketball."),
                "Q3": _yes_no_question("Is this an orange basketball?"),
            },
            2: {
                "Q1": _open_question("What color is this basketball?"),
                "Q2": _open_question("Describe the overall color of this basketball."),
                "Q3": _yes_no_question("Is this an orange basketball?"),
            },
            3: {
                "Q1": _open_question("What color is this basketball?"),
                "Q2": _open_question("Describe the overall color of this basketball."),
                "Q3": _yes_no_question("Is this an orange basketball?"),
            },
        },
    },
    "tennis_ball": {
        "sanity": {
            "Q1": _open_question("What color is this tennis ball?"),
            "Q2": _open_question("Describe the color of this tennis ball."),
            "Q3": _yes_no_question("Is this a yellow-green tennis ball?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What color is this tennis ball?"),
                "Q2": _open_question("Describe the color of this tennis ball."),
                "Q3": _yes_no_question("Is this a yellow-green tennis ball?"),
            },
            2: {
                "Q1": _open_question("What color is this tennis ball?"),
                "Q2": _open_question("Describe the color of this tennis ball."),
                "Q3": _yes_no_question("Is this a yellow-green tennis ball?"),
            },
            3: {
                "Q1": _open_question("What color is this tennis ball?"),
                "Q2": _open_question("Describe the color of this tennis ball."),
                "Q3": _yes_no_question("Is this a yellow-green tennis ball?"),
            },
        },
    },
    "piano_keyboard": {
        "sanity": {
            "Q1": _open_question("What color are the wide natural keys on this piano keyboard?"),
            "Q2": _open_question("Describe the colors of the keys on this piano keyboard."),
            "Q3": _yes_no_question("Are the wide natural keys white and the narrow raised keys black on this piano?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What color are the wide natural keys on this piano keyboard?"),
                "Q2": _open_question("Describe the color of the wide natural keys on this piano."),
                "Q3": _yes_no_question("Are the wide natural keys white on this piano?"),
            },
            2: {
                "Q1": _open_question("How many black keys are in each group on this piano keyboard?"),
                "Q2": _open_question("Describe the grouping pattern of the black keys on this keyboard."),
                "Q3": _yes_no_question("Are the black keys arranged in alternating groups of 2 and 3?"),
            },
            3: {
                "Q1": _open_question("What color are the raised sharp/flat keys on this piano keyboard?"),
                "Q2": _open_question("Describe the color of the narrow raised keys on this piano."),
                "Q3": _yes_no_question("Are the raised sharp/flat keys black on this piano?"),
            },
        },
    },
    "poker_cards": {
        "sanity": {
            "Q1": _open_question("What color is the heart symbol on these playing cards?"),
            "Q2": _open_question("Describe the colors of the suit symbols on these cards."),
            "Q3": _yes_no_question("Are the hearts red and spades black on these cards?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What color is the heart symbol on these playing cards?"),
                "Q2": _open_question("Describe the colors of the heart and spade symbols on these cards."),
                "Q3": _yes_no_question("Are the hearts red and spades black on these cards?"),
            },
            2: {
                "Q1": _open_question("What color is the diamond symbol on these playing cards?"),
                "Q2": _open_question("Describe the colors of the diamond and club symbols on these cards."),
                "Q3": _yes_no_question("Are the diamonds red and clubs black on these cards?"),
            },
            3: {
                "Q1": _open_question("How many red suit symbols are visible across these cards?"),
                "Q2": _open_question("Count the number of red-colored suit symbols on these playing cards."),
                "Q3": _yes_no_question("Is there at least one red suit symbol visible?"),
            },
        },
    },
}


EVAL_FASHION = {
    "ralph_lauren": {
        "sanity": {
            "Q1": _open_question("Which direction is the polo rider facing in this logo?"),
            "Q2": _open_question("Describe the direction the horseback rider faces in this emblem."),
            "Q3": _yes_no_question("Is the polo rider facing right?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("Which direction is the polo rider facing in this logo?"),
                "Q2": _open_question("Describe the direction the horseback rider faces."),
                "Q3": _yes_no_question("Is the polo rider facing right?"),
            },
            2: {
                "Q1": _open_question("Is the polo rider holding a mallet or polo stick?"),
                "Q2": _open_question("Describe what the rider is holding in their hand."),
                "Q3": _yes_no_question("Does the polo rider have a mallet?"),
            },
            3: {
                "Q1": _open_question("What animal is the rider sitting on in this logo?"),
                "Q2": _open_question("Describe the animal in this emblem."),
                "Q3": _yes_no_question("Is the rider sitting on a horse?"),
            },
        },
    },
    "burberry": {
        "sanity": {
            "Q1": _open_question("Which direction is the knight facing in this emblem?"),
            "Q2": _open_question("Describe the direction the equestrian knight faces."),
            "Q3": _yes_no_question("Is the knight facing left?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("Which direction is the knight facing in this emblem?"),
                "Q2": _open_question("Describe the direction the equestrian knight faces."),
                "Q3": _yes_no_question("Is the knight facing left?"),
            },
            2: {
                "Q1": _open_question("Is the knight carrying a flag or lance?"),
                "Q2": _open_question("Describe what the knight is holding."),
                "Q3": _yes_no_question("Does the knight have a flag?"),
            },
            3: {
                "Q1": _open_question("Is there a rider on the horse in this emblem?"),
                "Q2": _open_question("Describe who or what is on the horse."),
                "Q3": _yes_no_question("Is there a knight riding the horse?"),
            },
        },
    },
    "loewe": {
        "sanity": {
            "Q1": _open_question("How many L letters are in this Anagram logo?"),
            "Q2": _open_question("Count the L letters in this logo."),
            "Q3": _yes_no_question("Does this logo have four L letters?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("How many L letters are in this Anagram logo?"),
                "Q2": _open_question("Count the L letters in this logo."),
                "Q3": _yes_no_question("Does this logo have four L letters?"),
            },
            2: {
                "Q1": _open_question("Is this Anagram logo in its standard orientation or rotated?"),
                "Q2": _open_question("Describe the orientation of this logo."),
                "Q3": _yes_no_question("Is this logo in its standard upright orientation?"),
            },
            3: {
                "Q1": _open_question("Are the L letters in this logo overlapping or separated?"),
                "Q2": _open_question("Describe how the L letters are arranged relative to each other."),
                "Q3": _yes_no_question("Are the L letters overlapping and interlocking?"),
            },
        },
    },
    "chanel": {
        "sanity": {
            "Q1": _open_question("How are the two C's oriented in this logo - face-to-face or back-to-back?"),
            "Q2": _open_question("Describe the orientation of the two interlocking C's."),
            "Q3": _yes_no_question("Are the two C's back-to-back (openings facing outward)?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("How are the two C's oriented in this logo - face-to-face or back-to-back?"),
                "Q2": _open_question("Describe the orientation of the two C's."),
                "Q3": _yes_no_question("Are the two C's back-to-back (openings facing outward)?"),
            },
            2: {
                "Q1": _open_question("Are the C openings pointing left/right or up/down?"),
                "Q2": _open_question("Describe the direction the C openings face."),
                "Q3": _yes_no_question("Are the C openings pointing left and right (standard orientation)?"),
            },
            3: {
                "Q1": _open_question("Are the two C's the same size or different sizes?"),
                "Q2": _open_question("Describe the relative sizes of the two C's."),
                "Q3": _yes_no_question("Are the two C's the same size?"),
            },
        },
    },
    "lv": {
        "sanity": {
            "Q1": _open_question("In this LV monogram, which letter appears in front - L or V?"),
            "Q2": _open_question("Describe the layering of L and V in this monogram."),
            "Q3": _yes_no_question("Is the V in front of the L in this monogram?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("In this LV monogram, which letter appears in front - L or V?"),
                "Q2": _open_question("Describe the layering of L and V."),
                "Q3": _yes_no_question("Is the V in front of the L?"),
            },
            2: {
                "Q1": _open_question("What is the letter order in this monogram from left to right?"),
                "Q2": _open_question("Describe the order of letters in this monogram."),
                "Q3": _yes_no_question("Is the letter order L then V (left to right)?"),
            },
            3: {
                "Q1": _open_question("Are the L and V overlapping or separated?"),
                "Q2": _open_question("Describe how L and V are arranged."),
                "Q3": _yes_no_question("Are the L and V overlapping?"),
            },
        },
    },
    "gucci_gg": {
        "sanity": {
            "Q1": _open_question("Are the two G's facing opposite directions or the same direction?"),
            "Q2": _open_question("Describe the orientation of the two G letters."),
            "Q3": _yes_no_question("Are the two G's facing opposite directions (one upright, one inverted)?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("Are the two G's facing opposite directions or the same direction?"),
                "Q2": _open_question("Describe the orientation of the two G letters."),
                "Q3": _yes_no_question("Are the two G's facing opposite directions?"),
            },
            2: {
                "Q1": _open_question("Is this GG logo in its standard orientation or rotated?"),
                "Q2": _open_question("Describe the orientation of this double-G logo."),
                "Q3": _yes_no_question("Is this GG logo in its standard upright orientation?"),
            },
            3: {
                "Q1": _open_question("Are the two G's interlocking or separated?"),
                "Q2": _open_question("Describe how the two G letters are arranged."),
                "Q3": _yes_no_question("Are the two G's interlocking?"),
            },
        },
    },
    "ysl": {
        "sanity": {
            "Q1": _open_question("What is the letter order in this monogram from top to bottom?"),
            "Q2": _open_question("Describe the arrangement of letters in this YSL monogram."),
            "Q3": _yes_no_question("Is the letter order Y (top), S (middle), L (bottom)?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("What is the letter order in this monogram from top to bottom?"),
                "Q2": _open_question("Describe the arrangement of letters."),
                "Q3": _yes_no_question("Is the letter order Y (top), S (middle), L (bottom)?"),
            },
            2: {
                "Q1": _open_question("How many letters are in this monogram?"),
                "Q2": _open_question("Which letters are present in this monogram?"),
                "Q3": _yes_no_question("Does this monogram contain the letter S?"),
            },
            3: {
                "Q1": _open_question("Are the letters in this monogram interlocking or separated?"),
                "Q2": _open_question("Describe how the letters are arranged."),
                "Q3": _yes_no_question("Are the Y, S, L letters interlocking?"),
            },
        },
    },
    "celine": {
        "sanity": {
            "Q1": _open_question("How are the two C's oriented in this Triomphe logo - face-to-face or back-to-back?"),
            "Q2": _open_question("Describe the orientation of the two C's in this Triomphe pattern."),
            "Q3": _yes_no_question("Are the two C's face-to-face (openings facing inward toward each other)?"),
        },
        "mods": {
            1: {
                "Q1": _open_question("How are the two C's oriented - face-to-face or back-to-back?"),
                "Q2": _open_question("Describe the orientation of the two C's."),
                "Q3": _yes_no_question("Are the two C's face-to-face?"),
            },
            2: {
                "Q1": _open_question("Is this Triomphe logo in its standard orientation or rotated?"),
                "Q2": _open_question("Describe the orientation of this logo."),
                "Q3": _yes_no_question("Is this Triomphe logo in its standard orientation?"),
            },
            3: {
                "Q1": _open_question("Are the two C's the same size or different sizes?"),
                "Q2": _open_question("Describe the relative sizes of the two C's."),
                "Q3": _yes_no_question("Are the two C's the same size?"),
            },
        },
    },
}


def _get_domain(key, domain=None):
    if domain:
        return domain
    if key in EVAL_INDUSTRY:
        return "industry"
    if key in EVAL_FASHION:
        return "fashion"
    raise KeyError(f"Unknown key: {key}")


def get_questions(category_or_brand, mod_id=None, domain=None):
    """
    Return the sanity or counterfactual question block for a key.
    """
    resolved_domain = _get_domain(category_or_brand, domain)
    db = EVAL_INDUSTRY if resolved_domain == "industry" else EVAL_FASHION
    entry = db[category_or_brand]
    questions = entry["sanity"] if mod_id is None else entry["mods"][mod_id]
    return deepcopy(questions)


def build_question_payload(key, mod_id, question_type, input_mode, prime=True, tool_condition="raw"):
    """
    Return question-level prompt and scoring targets for the active v2 design.
    """
    if mod_id is None:
        question = get_questions(key, mod_id=None)[question_type]
        return {
            "prompt": _format_question_with_prior(question, key, mod_id=None, prime=prime),
            "correct": None,
            "biased": None,
            "evaluation_target": None,
            "bias_rationale": "",
            "mc_options": None,
            "correct_letter": None,
            "biased_letter": None,
            "prior_statement": prior_statement_for(key, mod_id=None),
            "question_design_version": QUESTION_DESIGN_VERSION,
        }

    design = QUESTION_DESIGNS[(key, mod_id)]
    question_spec = design["questions"][question_type]
    prior_statement = design["prior_statement"]

    if is_both_raw_pair_difference(input_mode, tool_condition):
        correct_value = "no" if question_type == "Q3" else "different"
        biased_value = "yes" if question_type == "Q3" else "same or canonical prior"
    elif question_type == "Q3":
        correct_value = "yes" if input_mode == "orig_only" else question_spec["correct"]
        biased_value = "yes" if input_mode == "orig_only" else question_spec["biased"]
    elif input_mode == "orig_only":
        correct_value = question_spec["biased"]
        biased_value = None
    else:
        correct_value = question_spec["correct"]
        biased_value = question_spec["biased"]

    mc_options = None
    correct_letter = None
    biased_letter = None
    if is_both_raw_pair_difference(input_mode, tool_condition) and question_type == "Q_MC":
        mc_prompt = _pair_difference_mc_prompt(key, mod_id, prime=prime)
        prompt = mc_prompt["prompt"]
        mc_options = mc_prompt["options"]
        correct_letter = mc_prompt["correct_letter"]
        biased_letter = mc_prompt["biased_letter"]
    elif question_type == "Q_MC":
        mc_prompt = build_mc_prompt(
            key=key,
            mod_id=mod_id,
            correct_value=correct_value,
            biased_value=biased_value,
            input_mode=input_mode,
            prime=prime,
            distractor_pool=question_spec.get("distractors"),
        )
        prompt = mc_prompt["prompt"]
        mc_options = mc_prompt["options"]
        correct_letter = mc_prompt["correct_letter"]
        biased_letter = mc_prompt["biased_letter"]
    elif is_both_raw_pair_difference(input_mode, tool_condition):
        prompt = _format_pair_question_with_prior(
            _pair_difference_open_question(key, mod_id, question_type),
            key,
            mod_id=mod_id,
            prime=prime,
        )
    else:
        prompt = _format_question_with_prior(question_spec["prompt"], key, mod_id=mod_id, prime=prime)

    bias_rationale = ""
    if biased_value is not None:
        if is_both_raw_pair_difference(input_mode, tool_condition):
            bias_rationale = (
                f"{prior_statement} A model relying on prior knowledge may ignore the pair difference "
                "and claim the target attribute is unchanged or canonical."
            )
        else:
            bias_rationale = (
                f"{prior_statement} A model relying on prior knowledge may answer "
                f"'{biased_value}' even when the visible target supports '{correct_value}'."
            )

    return {
        "prompt": prompt,
        "correct": correct_value,
        "biased": biased_value,
        "evaluation_target": EVALUATION_TARGETS.get((key, mod_id), question_spec.get("prompt", "")),
        "bias_rationale": bias_rationale,
        "mc_options": mc_options,
        "correct_letter": correct_letter,
        "biased_letter": biased_letter,
        "prior_statement": prior_statement,
        "question_design_version": QUESTION_DESIGN_VERSION,
        **both_raw_policy_fields(input_mode, tool_condition),
    }


def get_all_questions():
    """
    Return every question block as a flat list.
    """
    records = []
    for domain, db in (("industry", EVAL_INDUSTRY), ("fashion", EVAL_FASHION)):
        for key, entry in db.items():
            records.append({
                "domain": domain,
                "key": key,
                "mod_id": None,
                "type": "sanity",
                **deepcopy(entry["sanity"]),
            })
            for mod_id, questions in entry["mods"].items():
                records.append({
                    "domain": domain,
                    "key": key,
                    "mod_id": mod_id,
                    "type": "counterfactual",
                    **deepcopy(questions),
                })
    return records


def get_primed_question(key, qt, mod_id=None, prime=True):
    """
    Return a raw or prime-wrapped question for sanity or CF use.
    """
    if qt == "Q_MC":
        raise NotImplementedError("Q_MC must be constructed via build_mc_prompt()")
    if mod_id is not None and (key, mod_id) in QUESTION_DESIGNS:
        return build_question_payload(key, mod_id, qt, input_mode="cf_only", prime=prime)["prompt"]
    questions = get_questions(key, mod_id=mod_id)
    question = questions[qt]
    return _format_question_with_prior(question, key, mod_id=mod_id, prime=prime)


def print_summary():
    """
    Print a compact summary of the question library.
    """
    total_sanity = 0
    total_cf = 0

    print("=" * 70)
    print("VLM Bias Question Summary")
    print("=" * 70)

    for domain_name, db in (("Industry", EVAL_INDUSTRY), ("Fashion", EVAL_FASHION)):
        print(f"\n[{domain_name}]")
        for key, entry in db.items():
            total_sanity += 1
            total_cf += len(entry["mods"])
            print(f"- {key}: sanity + {len(entry['mods'])} mods")
            print(f"  sanity Q1: {entry['sanity']['Q1']}")
            for mod_id, questions in entry["mods"].items():
                print(f"  mod {mod_id} Q1: {questions['Q1']}")

    print(f"\nTotal templates: {total_sanity} sanity + {total_cf} CF = {total_sanity + total_cf}")


if __name__ == "__main__":
    print_summary()
