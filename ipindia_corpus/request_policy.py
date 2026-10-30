"""Versioned conservative request policy; source text never enters this function."""
import re
import unicodedata

from .grounded_answers import INJECTION, PERSONAL, TEMPORAL

VERSION = "archived-request-policy-v2"
CURRENT = re.compile(r"\b(?:present(?:ly)?|operative|enforceab\w*|still (?:legally |legally )?(?:valid|applicable)|up.to.date|as it stands|legal validity|kanoonan|aaj|abhi|filhaal)\b|इस\s+समय|अब\s+भी|मौजूदा|वर्तमान|लागू|प्रभावी|वैध", re.I)
ADVICE = re.compile(r"\b(?:should (?:we|my)|my (?:family|child|recipe|formulation)|meri|mera|mere|mujhe|chahiye|dosage|dose|prescribe|prescription)\b|क्या\s+मैं|मुझे|मेरे|मेरी|अपने\s+बच्चे|खुराक|इलाज|उपचार", re.I)
UNSAFE = re.compile(r"reveal\s+(?:your\s+)?system|change\s+(?:the\s+)?(?:crawler|host|schedule)|mark\s+every.*in_force|replace\s+the\s+source\s+hash|निर्देश.*अनदेखा", re.I)
AMBIGUOUS = re.compile(r"\b(?:legally controls|binding rule|binding legal|conflicting.*(?:versions?|dates?)|resolve.*conflict|unverified version|valid hai|lagu hai|effective date)\b|अप्रमाणित|बाध्यकारी|विरोधी", re.I)


def classify(question):
    if not isinstance(question, str) or not 2 <= len(question.strip()) <= 500:
        kind, reason = "ambiguous", "invalid_question_or_limit"
    else:
        q = unicodedata.normalize("NFKC", question).casefold()
        if INJECTION.search(q) or UNSAFE.search(q):
            kind, reason = "unsafe_request", "unsafe_instruction_request"
        elif PERSONAL.search(q) or ADVICE.search(q):
            kind, reason = "personal_advice", "personal_legal_or_medical_advice"
        elif TEMPORAL.search(q) or CURRENT.search(q):
            kind, reason = "current_status", "current_legal_status_unverified"
        elif AMBIGUOUS.search(q):
            kind, reason = "ambiguous", "intent_uncertain"
        else:
            kind, reason = "archived_research", None
    return {"version": VERSION, "kind": kind, "reason": reason, "allowed": reason is None}
