import re
from typing import List

# Map lowercase Greek letters to their English names so that queries written
# as words (e.g. "alpha") can match abstracts that use the Greek glyph (e.g.
# "α"), and vice versa. Greek letters are otherwise outside [a-zA-Z0-9\-] and
# would be silently dropped by the regex below.
_GREEK_LETTER_NAMES = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}

_GREEK_LETTER_RE = re.compile("[" + "".join(_GREEK_LETTER_NAMES) + "]")


def tokenize(text: str) -> List[str]:
    text = _GREEK_LETTER_RE.sub(lambda m: f" {_GREEK_LETTER_NAMES[m.group()]} ", text.lower())
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b", text)
