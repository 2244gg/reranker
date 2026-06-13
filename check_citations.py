"""Final cross-check of citations: tex \cite vs bib @entry keys."""
from pathlib import Path
import re

tex_files = list(Path("docs/latex").glob("*.tex"))
cites = set()
for f in tex_files:
    text = f.read_text(encoding="utf-8")
    for m in re.findall(r"\\cite\{([^}]+)\}", text):
        for k in m.split(","):
            cites.add(k.strip())

bib_text = Path("docs/latex/paper.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))

print(f"Cited keys ({len(cites)}):", sorted(cites))
print()
print(f"Bib keys ({len(bib_keys)}):", sorted(bib_keys))
print()
orphans = cites - bib_keys
unused = bib_keys - cites
print(f"Orphan cites (in tex but not in bib): {orphans if orphans else 'NONE — clean'}")
print(f"Unused bib entries: {unused if unused else 'NONE'}")
