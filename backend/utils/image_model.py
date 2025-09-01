from PIL import Image
from tensorflow.keras.models import load_model
import threading

import numpy as np

# ----- klasy obrazu (3) -----
IMAGE_MAPPER = {
    0: "healthy/doubtful",
    1: "moderate",
    2: "severe",
}

# ----- odwzorowanie 3 klas → 5-stopniowe KL -----
IMAGE_TO_KL = {
    "healthy/doubtful": "KL1",  # możesz dać "KL0" jeśli w Twoich danych to faktycznie zdrowi
    "moderate":         "KL3",
    "severe":           "KL4",
}

# ----- labelki i ryzyko (masz już wyżej) -----
KL_TO_GRADE = {
    "KL0": "Grade 0 - none",
    "KL1": "Grade 1 - doubtful",
    "KL2": "Grade 2 - minimal",
    "KL3": "Grade 3 - moderate",
    "KL4": "Grade 4 - severe",
}
RISK_FROM_KL = {
    "KL0": "Very low",
    "KL1": "Low",
    "KL2": "Moderate",
    "KL3": "High",
    "KL4": "Very high",
}

# ----- kosmetyka kafelka (kolory + krótkie tipy) -----
COLOR_BY_KL = {
    "KL0": "#2e7d32",  # green
    "KL1": "#558b2f",
    "KL2": "#f9a825",  # amber
    "KL3": "#ef6c00",  # orange
    "KL4": "#c62828",  # red
}
TIPS_BY_KL = {
    "KL0": [        
        "Continue low-impact activities.",
        "Maintain proper warm-up and exercise technique."
    ],
    "KL1": [
        "Monitor symptoms with heavier loads.",
        "Incorporate range-of-motion exercises and light strengthening."
    ],
    "KL2": [
        "Consider knee-specific physical therapy.",
        "Prefer low-impact activities (cycling, swimming)."
    ],
    "KL3": [
        "Consult a treatment plan with an orthopedist/physiotherapist.",
        "Adjust loads and enhance stabilization."
    ],
    "KL4": [
        "Specialist consultation required.",
        "Ensure pain control as recommended by your doctor."
    ],
}


_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_PATH = 'D:/semestr_10/master_thesis/medical_chatbot/notebooks/models/radimagenet_best_model.h5'

def _get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:  # double-checked locking
                _MODEL = load_model(_MODEL_PATH)
    return _MODEL

# def build_image_result_card(kl_code: str, confidence: float = None, probs: dict | None = None) -> dict:
#     """Zwraca kartę wyniku w tym samym stylu, co tekstowa (spójne pola)."""
#     grade_label = KL_TO_GRADE.get(kl_code, "Unknown")
#     try:
#         grade_num = int(kl_code.replace("KL",""))
#     except Exception:
#         grade_num = None

#     card = {
#         "mode": "image",
#         "kl_code": kl_code,                 # np. "KL2"
#         "grade_label": grade_label,         # np. "Grade 2 - minimal"
#         "grade_num": grade_num,             # np. 2
#         "risk_label": RISK_FROM_KL.get(kl_code, "Unknown"),
#         "confidence": confidence,           # np. 0.87
#     }
#     if probs:
#         card["class_probabilities"] = probs
#     return card

def build_image_result_card(kl_code: str, confidence: float = None, probs: dict | None = None) -> dict:
    # bezpieczny fallback:
    if not kl_code:
        kl_code = "KL3"

    grade_label = KL_TO_GRADE.get(kl_code, "Unknown")
    try:
        grade_num = int(kl_code.replace("KL",""))
    except Exception:
        grade_num = None

    card = {
        "mode": "image",
        "source": "MRI",                    # dla frontu, jeśli chcesz odróżniać
        "kl_code": kl_code,                 # np. "KL3"
        "grade_label": grade_label,         # "Grade 3 - moderate"
        "grade_title": "MRI-based assessment",
        "grade_num": grade_num,             # 3
        "risk_label": RISK_FROM_KL.get(kl_code, "Unknown"),
        "color": COLOR_BY_KL.get(kl_code, "#666"),
        "confidence": float(confidence) if confidence is not None else None,
        # WOMAC nie dotyczy obrazu → pomijamy
        "tips": TIPS_BY_KL.get(kl_code, []),
    }
    if probs:
        # oczekujemy { "KLx": p, ... } – upewnij się że klucze to KL*
        card["class_probabilities"] = {
            str(k): float(v) for k, v in probs.items()
        }
    return card

def adjust_image(image):
    img = Image.open(image.stream)

    # converting to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # adjusting size of the image to make prediction
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image_classification(image):
    model = _get_model()
    return model.predict(image)