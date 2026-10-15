from src.labels import LETTERS, NUM_CLASSES, CHAR_TO_INDEX

print("Classes:", NUM_CLASSES)
assert NUM_CLASSES == 33
assert len(set(CHAR_TO_INDEX)) == 33, "duplicate character found"
print("Index 0:", LETTERS[0])
print("OK")