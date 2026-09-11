import re
from typing import Dict, List

# A lexicon mapping Marathi municipal keywords to English standard terms
MARATHI_LEXICON: Dict[str, str] = {
    # Solid Waste
    "कचरा": "garbage",
    "घाण": "dirt",
    "साफसफाई": "cleaning",
    "डस्टबिन": "dustbin",
    "दुर्गंधी": "bad smell",
    "स्वच्छता": "cleanliness",
    "कुंडी": "bin",

    # Water & Infrastructure Pipelines
    "पाणी": "water",
    "गळती": "leakage",
    "नळ": "tap",
    "पाणीपुरवठा": "water supply",
    "दूषित": "contaminated",
    "पाइप": "pipe",
    "पाइपलाइन": "pipeline",
    "फुटली": "burst",
    "फुटला": "burst",

    # Non-water substances & utilities (Out of Scope / Disambiguation)
    "तेल": "oil",
    "तेलाची": "oil",
    "तेलाच्या": "oil",
    "ऑईल": "oil",
    "गॅस": "gas",
    "केमिकल": "chemical",
    "रसायन": "chemical",
    "पेट्रोल": "petrol",
    "डिझेल": "diesel",

    # Roads
    "रस्ता": "road",
    "रस्ते": "roads",
    "खड्डा": "pothole",
    "खड्डे": "potholes",
    "डांबरीकरण": "asphalting",
    "फुटपाथ": "footpath",

    # Drainage
    "गटार": "drain",
    "गटारे": "drains",
    "नाला": "drainage",
    "तुंबले": "blocked",
    "पाणी साचले": "waterlogging",
    "सांडपाणी": "wastewater",

    # Street Lighting & Electrical
    "लाईट": "light",
    "दिवा": "lamp",
    "पथकर": "streetlamp",
    "अंधार": "darkness",
    "विद्युत": "electricity",
    "वीज": "electricity",
    "करंट": "electric current",

    # Animals
    "भटके": "stray",
    "कुत्रे": "dogs",
    "डुक्कर": "pig",
    "मोकाट": "stray",

    # General / Admin
    "तक्रार": "complaint",
    "समस्या": "problem",
    "महानगरपालिका": "municipal corporation",
    "अधिकारी": "officer",
    "काम": "work",
    "त्रास": "trouble",
    "लवकर": "early",
    "तातडीने": "urgently"
}

def normalize_marathi_text(text: str) -> str:
    """
    Normalizes Marathi text by appending English equivalent terms for matched 
    Marathi municipal keywords, acting as query expansion for TF-IDF.
    """
    if not text:
        return text

    normalized_text = text
    
    # Sort by length descending to avoid partial matches
    sorted_keys = sorted(MARATHI_LEXICON.keys(), key=len, reverse=True)
    
    matched_english_terms = []
    for word in sorted_keys:
        if word in text:
            matched_english_terms.append(MARATHI_LEXICON[word])
            
    if matched_english_terms:
        # Append matched English terms to the original text
        expansion = " ".join(matched_english_terms)
        normalized_text = f"{text} {expansion}"
        
    return normalized_text
