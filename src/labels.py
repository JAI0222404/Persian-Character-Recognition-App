# (character, name, transliteration)
LETTERS = [
    ("ا", "Alef", "a"), ("آ", "Alef-madda", "aa"), ("ب", "Be", "b"),
    ("پ", "Pe", "p"), ("ت", "Te", "t"), ("ث", "Se (3 dots)", "s"),
    ("ج", "Jim", "j"), ("چ", "Che", "ch"), ("ح", "He (jimi)", "h"),
    ("خ", "Khe", "kh"), ("د", "Dal", "d"), ("ذ", "Zal", "z"),
    ("ر", "Re", "r"), ("ز", "Ze", "z"), ("ژ", "Zhe", "zh"),
    ("س", "Sin", "s"), ("ش", "Shin", "sh"), ("ص", "Sad", "s"),
    ("ض", "Zad", "z"), ("ط", "Ta", "t"), ("ظ", "Za", "z"),
    ("ع", "Ayn", "'"), ("غ", "Ghayn", "gh"), ("ف", "Fe", "f"),
    ("ق", "Qaf", "gh"), ("ک", "Kaf", "k"), ("گ", "Gaf", "g"),
    ("ل", "Lam", "l"), ("م", "Mim", "m"), ("ن", "Nun", "n"),
    ("و", "Vav", "v"), ("ه", "He (do-chashm)", "h"), ("ی", "Ye", "y"),
]

CHARS = [c for c, _, _ in LETTERS]
NUM_CLASSES = len(LETTERS)          # 33
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHARS)}
INDEX_TO_INFO = {i: info for i, info in enumerate(LETTERS)}