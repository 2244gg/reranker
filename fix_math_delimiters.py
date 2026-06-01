"""Convert LaTeX-style math delimiters to dollar-sign delimiters
for standard markdown rendering.

\[ ... \]   -> $$ ... $$    (display math)
\( ... \)   -> $ ... $       (inline math)
"""
from pathlib import Path
import re

p = Path("docs/formulas_reference.md")
text = p.read_text(encoding="utf-8")

# Block math: \[\n  body  \n\]   ->   $$\n  body  \n$$
# Match across newlines.
text = re.sub(
    r"\\\[\s*\n",
    "$$\n",
    text,
)
text = re.sub(
    r"\n\s*\\\]",
    "\n$$",
    text,
)

# Inline math: \( body \)  ->  $body$
text = re.sub(r"\\\(", "$", text)
text = re.sub(r"\\\)", "$", text)

p.write_text(text, encoding="utf-8")

# Verify no leftover \[ \] \( \)
remaining = re.findall(r"\\\[|\\\]|\\\(|\\\)", text)
print(f"converted. remaining old delimiters: {len(remaining)}")

# Count new ones for sanity
print(f"new $$ blocks: {text.count('$$') // 2}")
print(f"new $ inline: roughly {text.count('$') - 2 * (text.count('$$') // 2)}")
