"""Versioned conservative request policy; source text never enters this function."""
import re
import unicodedata

from .grounded_answers import INJECTION, PERSONAL, TEMPORAL

VERSION = "archived-request-policy-v3"
CURRENT = re.compile(r"\b(?:present(?:ly)?|operative|enforceab\w*|still (?:legally |legally )?(?:valid|applicable)|up.to.date|as it stands|legal validity|kanoonan|aaj|abhi|filhaal)\b|इस\s+समय|अब\s+भी|मौजूदा|वर्तमान|लागू|प्रभावी|वैध", re.I)
ADVICE = re.compile(r"\b(?:should (?:we|my)|my (?:family|child|recipe|formulation)|meri|mera|mujhe|chahiye|mere\s+(?:liye|patent|invention|business|recipe|formulation|bache|bachche|parivar)|mere\s+(?:product|utpad|idea|company|case)\s+(?:ko|ka|ki|ke|par|mein|liye)|dosage|dose|prescribe|prescription)\b|क्या\s+मैं|मुझे|मेरे|मेरी|अपने\s+बच्चे|खुराक", re.I)
# A treatment term alone is not personalized advice when the user asks about
# an identified archived provision. Personal markers and clinical instructions
# still take precedence; adding a section number cannot neutralize them.
LEGAL_REFERENCE = re.compile(r"\b(?:sections?|rules?)\s+\d|धारा\s+[0-9०-९]|नियम\s+[0-9०-९]", re.I)
ARCHIVED_INQUIRY = re.compile(r"क्या.{0,30}(?:कहा|प्रावधान|परिभाषा|वर्णन)|(?:कौन|किस).{0,35}(?:प्रावधान|जानकारी)|(?:उद्धरण|उद्धृत|परिभाषा)", re.I)
MEDICAL_TOPIC = re.compile(r"इलाज|उपचार", re.I)
CLINICAL_REQUEST = re.compile(r"(?:सलाह|सुझाव)\s*(?:दें|दो|बताएं)|सुझा(?:एं|इए)|कौन\s*(?:सा|सी).{0,50}(?:इलाज|उपचार|दवा)|(?:इलाज|उपचार|दवा).{0,40}(?:बताइ|बताए|बताओ|करें|करूँ|लेना|करना)", re.I)
UNSAFE = re.compile(r"reveal\s+(?:your\s+)?system|change\s+(?:the\s+)?(?:crawler|host|schedule)|mark\s+every.*in_force|replace\s+the\s+source\s+hash|निर्देश.*अनदेखा", re.I)
AMBIGUOUS = re.compile(r"\b(?:legally controls|binding rule|binding legal|conflicting.*(?:versions?|dates?)|resolve.*conflict|unverified version|valid hai|lagu hai|effective date)\b|अप्रमाणित|बाध्यकारी|विरोधी", re.I)


def classify(question):
    if not isinstance(question, str) or not 2 <= len(question.strip()) <= 500:
        kind, reason = "ambiguous", "invalid_question_or_limit"
    else:
        q = unicodedata.normalize("NFKC", question).casefold()
        if INJECTION.search(q) or UNSAFE.search(q):
            kind, reason = "unsafe_request", "unsafe_instruction_request"
        elif (PERSONAL.search(q) or ADVICE.search(q) or CLINICAL_REQUEST.search(q)
              or (MEDICAL_TOPIC.search(q) and not (LEGAL_REFERENCE.search(q) and ARCHIVED_INQUIRY.search(q)))):
            kind, reason = "personal_advice", "personal_legal_or_medical_advice"
        elif TEMPORAL.search(q) or CURRENT.search(q):
            kind, reason = "current_status", "current_legal_status_unverified"
        elif AMBIGUOUS.search(q):
            kind, reason = "ambiguous", "intent_uncertain"
        else:
            kind, reason = "archived_research", None
    return {"version": VERSION, "kind": kind, "reason": reason, "allowed": reason is None}
