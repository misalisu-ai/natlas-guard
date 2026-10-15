import re
from typing import Tuple, List, Optional
from contracts import SafetyRoute, GateDecision, NormalizedQuery
from normalizer import normalize_hausa_utterance

# ---------------------------------------------------------
# 1. Linguistic Patterns & Multi-word Stems
# ---------------------------------------------------------

# Hausa Negation prefixes that invert symptom reporting
NEGATION_PREFIXES = [
    r"\bba\s+na\s+(jin|fama\s+da)\b",
    r"\bba\s+ya\s+(jin|yi)\b",
    r"\bban\s+da\b",
    r"\bbabu\s+alamar\b",
    r"\bbabu\b",
    r"\bba\s+wata\b"
]

# Priority 1: Emergency & Acute Danger Signs
# Calibrated for normalized routing representations (ASCII-folded, lowercase, stripped punctuation)
EMERGENCY_STEMS = [
    # Cardiac & Acute Chest (handles inverted predicates, burning, crushing metaphors)
    r"\b(ciwo|zafi|kuna|matsa|radadi)\w*\s+(?:(?:\w+\s+){0,4})?kirji\w*",
    r"\bkirji\w*\s+(?:(?:\w+\s+){0,4})?(ciwo|zafi|kuna|matsa|radadi)\b",
    r"\bbugun\s+zuciya\b",
    r"\bzuciya\w*\s+(?:(?:\w+\s+){0,4})?bugawa\b",
    r"\bzafi\s+kamar\s+an\s+dora\s+dutse\b",
    
    # Respiratory Distress & Cyanosis (handles long-distance predicates and 'baya numfashi')
    r"\b(numfashi|nishi)\s+sama\s+sama\b",
    r"\b(kasa|gaza|wahalar|daina|ba|baya|ya\s+kasa)\s+[^.?!]*?numfashi\b",
    r"\bshake\w*\s+a\s+makogwaro\b",
    r"\bmakogwaro\w*\s+[^.?!]*?shake\b",
    r"\bhanci\w*\s+[^.?!]*?budewa\s+da\s+karfi\b",
    r"\blebon\w*\s+[^.?!]*?(zama\s+baki|baki)\b",
    
    # Severe Bleeding, Preterm Labor & Obstetric Crises (handles multi-word descriptions)
    r"\bzubar\s+[^.?!]*?jini\b",
    r"\bjini\s+[^.?!]*?(tsayawa|fita|zuba|tsanani|yawa)\b",
    r"\b(nakuda|ciwon\s+ciki)\s+(?:(?:\w+\s+){0,3})?kafin\s+lokaci\b",
    r"\b(ciki|goyon\s+ciki)\s+(?:(?:\w+\s+){0,4})?zubar\s+jini\b",
    r"\brauni\s+(?:(?:\w+\s+){0,4})?jini\b",
    
    # Neurological, Collapse & Loss of Consciousness
    r"\b(suma|sumewa|sume)\b",
    r"\byanke\s+jiki\b",
    r"\bfarfadiya\b",
    r"\brashin\s+gane\s+(mutane|inda\s+yake)\b",
    r"\bbaki\w*\s+ya\s+karkace\b",
    r"\bjiri\s+(?:(?:\w+\s+){0,3})?faduwa\b",
    
    # Severe Dehydration, Acute Pediatric Signs & Toxins / Envenomation
    r"\bzawo\s+(?:da\s+)?amai\b",
    r"\bzawayar\s+jini\b",
    r"\bidanu\w*\s+(?:(?:\w+\s+){0,2})?fada\s+ciki\b",
    r"\bfata\w*\s+(?:(?:\w+\s+){0,4})?bata\s+komawa\b",
    r"\bdaina\s+shan\s+mama\b",
    r"\bsha\s+(?:(?:\w+\s+){0,2})?(guba|acid|maganin\s+wari|ruwan\s+batir)\b",
    r"\b(kare|maciji)\s+ya\s+(ciji|sara)\b",
    r"\bkunar\s+wuta\b"
]

# Priority 2: Inappropriate Prescription & Dosage Seeking
# Captures dialectal spelling (siya vs saya), purchasing intent, and OTC pharmacy requests
PRESCRIPTION_STEMS = [
    r"\bwan[ei]\s+magani\b",
    r"\bmaganin\s+(me|wan[ei])\b",
    r"\bina\s+son\s+magani\b",
    r"\bbani\s+(sunan\s+)?magani\b",
    r"\bkwayar\s+magani\b",
    r"\badadin\s+[^.?!]*?sha\b",
    r"\bkwaya\s+(nawa|biyu|uku|hudu)\b",
    r"\bsha\s+[^.?!]*?(sau\s+\w+|kwaya\s+\w+)\b",
    r"\ballura\w*\s+(ta\s+|na\s+)?me\b",
    r"\b(zan\s+)?s[ai]y[ae]\s+[^.?!]*?magani\b",
    r"\bmagani\s+[^.?!]*?s[ai]y[ae]\b",
    r"\b(s[ai]y[ae]|bawa|sha)\s+magani\b",
    r"\b(a\s+)?(kemis|chemist|shagon\s+magani)\b",
    r"\b(syrup|antibiotic|paracetamol|panadol|ampicillin|ciprofloxacin)\b"
]

# Priority 3: Adversarial / Unsafe Self-Treatment
ADVERSARIAL_STEMS = [
    r"\b(daina|barin)\s+maganin\s+asibiti\b",
    r"\b(madadin|ya\s+fi)\s+maganin\s+asibiti\b",
    r"\bhadin\s+maganin\s+gargajiya\s+da\s+(na\s+)?asibiti\b",
]

# Priority 4: Under-Specified / Ambiguous Presentations
AMBIGUOUS_STEMS = [
    r"^\s*ciwon\s+ciki\s*$",
    r"^\s*kaina\s+na\s+ciwo\s*$",
    r"^\s*ina\s+jin\s+kasala\s*$",
    r"^\s*jikina\s+yana\s+min\s+wani\s+iri\s*$",
    r"^\s*yaro\s+yana\s+kuka\s*$"
]

# ---------------------------------------------------------
# 2. Clinically Audited Deterministic Responses
# ---------------------------------------------------------

RESPONSES = {
    SafetyRoute.EMERGENCY_ESCALATION: (
        "GARGADI NA GAGGAWA: Wadannan alamomi na nuni da matsala mai tsanani "
        "da ke bukatar kulawar likita nan take. Don Allah hanzarta zuwa asibiti ko "
        "cibiyar lafiya mafi kusa ba tare da bata lokaci ba. Kada ka nemi shan wani magani a gida."
    ),
    SafetyRoute.PRESCRIPTION_INTERCEPT: (
        "SANARWA KAN MAGANI: A matsayina na mataimaki na fasaha, ba ni da ikon rubuta "
        "ko ba da shawarar takamaiman sunan magani ko adadin kwayoyin da za a sha. "
        "Don Allah ziyarci kwararren likita ko masanin magunguna (pharmacist) a cibiyar lafiya "
        "don samun cikakken bincike da maganin da ya dace da lafiyarka."
    ),
    SafetyRoute.SAFE_ABSTENTION: (
        "FAHIMTA MAI AMINCI: Ba shi da aminci a daina shan magungunan da likita ya rubuta, "
        "ko hada maganin asibiti da na gargajiya ba tare da izinin likita ba. "
        "Don Allah tattauna da jami'in lafiyarka kafin yin wani canji ga tsarin shan maganinka."
    ),
    SafetyRoute.UNCERTAIN_AMBIGUOUS: (
        "NEMAN KARIN BAYANI: Bayanin da ka bayar yana da karanci kwarai don fahimtar hakikanin abin da ke damunka. "
        "Kuna iya bayyana tsawon lokacin da aka kwashe ana jin wannan yanayi, ko kuma ziyarci cibiyar lafiya "
        "mafi kusa idan kana jin ciwo mai tsanani."
    )
}

# ---------------------------------------------------------
# 3. Decision Logic with Precedence & Negation
# ---------------------------------------------------------

def check_negation(text: str, match_start: int) -> bool:
    """
    Checks if a symptom match is governed by a preceding negation within a 4-word window.
    """
    window = text[max(0, match_start - 35):match_start]
    for neg in NEGATION_PREFIXES:
        if re.search(neg, window):
            return True
    return False


def evaluate_safety_gate(query: NormalizedQuery) -> GateDecision:
    text = query.routing_representation
    
    # 1. Check Priority 1: Emergency Stems
    for pat in EMERGENCY_STEMS:
        match = re.search(pat, text)
        if match:
            if check_negation(text, match.start()):
                # Negated emergency symptom (e.g. "Ba na jin ciwon kirji") -> Do not escalate
                return GateDecision(
                    route=SafetyRoute.ALLOW_RAG,
                    deterministic_response="",
                    confidence=0.85,
                    trigger_rule=pat,
                    negation_detected=True,
                    metadata={"note": "Emergency symptom detected but negated"}
                )
            return GateDecision(
                route=SafetyRoute.EMERGENCY_ESCALATION,
                deterministic_response=RESPONSES[SafetyRoute.EMERGENCY_ESCALATION],
                confidence=1.0,
                trigger_rule=pat,
                negation_detected=False
            )
            
    # 2. Check Priority 2: Prescription Stems
    for pat in PRESCRIPTION_STEMS:
        match = re.search(pat, text)
        if match:
            return GateDecision(
                route=SafetyRoute.PRESCRIPTION_INTERCEPT,
                deterministic_response=RESPONSES[SafetyRoute.PRESCRIPTION_INTERCEPT],
                confidence=0.98,
                trigger_rule=pat
            )
            
    # 3. Check Priority 3: Adversarial / Unsafe Alternatives
    for pat in ADVERSARIAL_STEMS:
        match = re.search(pat, text)
        if match:
            return GateDecision(
                route=SafetyRoute.SAFE_ABSTENTION,
                deterministic_response=RESPONSES[SafetyRoute.SAFE_ABSTENTION],
                confidence=0.95,
                trigger_rule=pat
            )
            
    # 4. Check Priority 4: Ambiguous / Insufficient Symptom Presentation
    for pat in AMBIGUOUS_STEMS:
        if re.search(pat, text):
            return GateDecision(
                route=SafetyRoute.UNCERTAIN_AMBIGUOUS,
                deterministic_response=RESPONSES[SafetyRoute.UNCERTAIN_AMBIGUOUS],
                confidence=0.90,
                trigger_rule=pat
            )

    # 5. Priority 5: Safe to process with Grounded RAG
    return GateDecision(
        route=SafetyRoute.ALLOW_RAG,
        deterministic_response="",
        confidence=1.0,
        trigger_rule=None
    )