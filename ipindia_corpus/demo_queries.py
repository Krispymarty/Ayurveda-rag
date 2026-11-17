"""Bounded Hindi legal-term mapping for demo retrieval, never source rewriting.

No model expansion, invented provision IDs or language-based provider switching.
Unknown Hindi remains on the existing semantic path. English is byte-preserved.
"""
import re
import unicodedata

VERSION = "hindi-legal-terms-v1"
TERMS = {
    "पेटेंट अधिनियम": "Patents Act", "पेटेन्ट अधिनियम": "Patents Act",
    "पूर्ण पेटेंट विनिर्देश": "complete patent specification",
    "पेटेंट विनिर्देश": "patent specification", "पूर्ण विनिर्देश": "complete specification",
    "पारंपरिक ज्ञान": "traditional knowledge", "परम्परागत ज्ञान": "traditional knowledge",
    "आविष्कारक कदम": "inventive step", "आविष्कारशील कदम": "inventive step",
    "ज्ञात पदार्थों": "known substances", "साधारण मिश्रण": "mere admixture",
    "चिकित्सीय": "therapeutic", "औषधीय": "medicinal", "उपचार": "treatment",
    "इलाज": "treatment", "प्रक्रियाओं": "processes", "प्रक्रिया": "process",
    "परिभाषा": "definition", "पेटेंट": "patent", "पेटेन्ट": "patent",
    "अधिनियम": "Act", "धारा": "section", "धाराओं": "sections",
    "नियम": "rule", "नियमों": "rules", "मिश्रण": "mixture",
    "विनिर्देश": "specification", "विनिर्देशन": "specification",
    "विरोध": "opposition", "अपवाद": "exceptions", "पंजीकरण": "registration",
    "व्यापार चिह्न": "Trade Marks", "ट्रेड मार्क": "Trade Marks",
}
PATTERN = re.compile(r"(?<![\u0900-\u097f])(?:" + "|".join(re.escape(k) for k in sorted(TERMS, key=len, reverse=True)) + r")(?![\u0900-\u097f])")


def retrieval_query(question):
    if not re.search(r"[\u0900-\u097f]", question):
        return question
    normalized = unicodedata.normalize("NFKC", question)
    normalized = ''.join(str(unicodedata.digit(c)) if c.isdecimal() else c for c in normalized)
    mapped, count = PATTERN.subn(lambda match: TERMS[match[0]], normalized)
    if not count:
        return question
    # Remaining untranslated Hindi cannot be English FTS terms. The original
    # complete question still reaches both model passes without modification.
    query = re.sub(r"[\u0900-\u097f]+", " ", mapped)
    return re.sub(r"\s+", " ", query).strip()[:500]
