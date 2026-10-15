from pathlib import Path
import collections

root = Path(".")  # change to the dataset folder
counts = collections.Counter(p.parent.name for p in root.rglob("*")
                             if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
print(len(counts), "folders")
for name, n in sorted(counts.items())[:60]:
    print(name, n)