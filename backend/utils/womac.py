from typing import Dict, List, Tuple
import numpy as np

from utils.text_model import predict_class, MODEL_CLASSES
from config.constant import (
    GRADE_RANK,
    WOMAC_LABELS,
    WOMAC_SEVERITY_THRESHOLDS,
    LABEL_TO_GRADE,
    GRADE_TO_LABEL,
    DEFAULT_WOMAC_WEIGHT,
    DEFAULT_SCALE_MAX,
    QUESTIONS,
)

from utils.text_model import predict_class, MODEL_CLASSES
# jeśli w UI używasz „Grade …”, a model ma „KLx”, przygotuj translację:
KL_TO_GRADE = {
    "KL0": "Grade 0 - none",
    "KL1": "Grade 1 - doubtful",
    "KL2": "Grade 2 - minimal",
    "KL3": "Grade 3 - moderate",
    "KL4": "Grade 4 - severe",
}
GRADE_TO_KL = {v: k for k, v in KL_TO_GRADE.items()}

def _to_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _scale_questions() -> List[dict]:
    return [q for q in QUESTIONS.get("womac", []) if q.get("type") == "scale"]

def _legacy_by_id() -> Dict[str, str]:
    return {q["id"]: q["text"] for q in _scale_questions() if "id" in q and "text" in q}

def _get_resp(responses: Dict, qid: str, legacy_text: str, max_val: int) -> int:
    val = responses.get(qid)
    if val is None and legacy_text:
        val = responses.get(legacy_text)
    v = _to_int(val, default=0)
    if v < 0:
        v = 0
    if v > max_val:
        v = max_val
    return v

def calculate_womac_score(responses: Dict) -> Tuple[int, int]:
    legacy_map = _legacy_by_id()
    total = 0
    max_total = 0
    for q in _scale_questions():
        qid = q["id"]
        qmax = int(q.get("max", DEFAULT_SCALE_MAX))
        qtext = legacy_map.get(qid, "")
        total += _get_resp(responses, qid, qtext, qmax)
        max_total += qmax
    return total, max_total

def normalize_womac(score: int, max_score: int) -> float:
    if max_score <= 0:
        return 0.0
    return (score / max_score) * 100.0

def womac_severity_label(normalized_score: float) -> str:
    for threshold, label in WOMAC_SEVERITY_THRESHOLDS:
        if normalized_score <= threshold:
            return label
    # Bezpieczny fallback
    return WOMAC_LABELS[-1]

def _grade_rank_index(grade: str, default: int = len(GRADE_RANK) - 1) -> int:
    try:
        return GRADE_RANK.index(grade)
    except ValueError:
        return default

def _closest_grade(target_grade: str, available_grades: List[str]) -> str:
    """
    Znajdź najbliższą klasę (po pozycji w GRADE_RANK) w zbiorze dostępnych klas.
    """
    t_idx = _grade_rank_index(target_grade)
    best = available_grades[0]
    best_d = 1e9
    for g in available_grades:
        d = abs(_grade_rank_index(g) - t_idx)
        if d < best_d:
            best, best_d = g, d
    return best

def _align_model_classes(text_model_classes: List[str]) -> List[str]:
    return list(text_model_classes)

def combine_probs(
    womac_label: str,
    text_probs,
    womac_weight: float = DEFAULT_WOMAC_WEIGHT,
):
    import numpy as np
    text_probs = np.asarray(text_probs, dtype=float).ravel()
    model_classes_local = list(MODEL_CLASSES)  # np. ["KL0","KL1",...]

    if text_probs.size == 0:
        # sam WOMAC -> mapujemy do najbliższego KL
        target_grade = LABEL_TO_GRADE[womac_label]  # "Grade 2 - minimal"
        target_kl = GRADE_TO_KL.get(target_grade, "KL2")
        vec = np.zeros(len(model_classes_local), dtype=float)
        if target_kl in model_classes_local:
            vec[model_classes_local.index(target_kl)] = 1.0
        return vec, model_classes_local

    s = text_probs.sum()
    if s > 0:
        text_probs = text_probs / s

    target_grade = LABEL_TO_GRADE[womac_label]  # "Grade x - ..."
    target_kl = GRADE_TO_KL.get(target_grade)
    if target_kl not in model_classes_local:
        # najbliższy indeksowo (0..4)
        # wydłub cyfrę z Grade:
        import re
        m = re.search(r"Grade\s+(\d)", target_grade)
        idx = int(m.group(1)) if m else 2
        target_kl = f"KL{idx}"

    womac_vec = np.zeros(len(model_classes_local), dtype=float)
    womac_vec[model_classes_local.index(target_kl)] = 1.0

    combined = womac_weight * womac_vec + (1.0 - womac_weight) * text_probs
    return combined, model_classes_local


# def calculate_womac(response: Dict, womac_weight: float = DEFAULT_WOMAC_WEIGHT) -> str:
#     """
#     Główna funkcja: oblicz wynik WOMAC, znormalizuj do %, przypisz etykietę,
#     połącz z predykcją modelu tekstowego i zwróć końcową etykietę (string).
#     """
#     womac_score, max_score = calculate_womac_score(response)
#     normalized = normalize_womac(womac_score, max_score)
#     womac_label = womac_severity_label(normalized)

#     text_probs = predict_class(response)  # spodziewamy się wektora o długości == len(MODEL_CLASSES)
#     combined_probs, class_list = combine_probs(womac_label, text_probs, womac_weight)

#     idx = int(np.argmax(combined_probs))
#     chosen_kl  = class_list[idx]
#     combined_grade = KL_TO_GRADE.get(chosen_kl, "Grade 2 - minimal")
#     # Spróbuj zmapować na WOMAC label; jeśli nieznana klasa, fallback do WOMAC z ankiety
#     # combined_label = GRADE_TO_LABEL.get(combined_grade, womac_label)
#     combined_label = combined_grade

#     # Debug (opcjonalnie — możesz podpiąć pod logger)
#     # print(f"WOMAC raw: {womac_score}/{max_score} -> {normalized:.2f}% -> {womac_label}")
#     # print(f"Model classes used: {class_list} (len={len(class_list)})")
#     # print(f"Text probs: {np.asarray(text_probs).ravel()}")
#     # print(f"Combined:  {combined_probs} -> {combined_grade} -> {combined_label}")

#     return combined_label

def calculate_womac(response: Dict, womac_weight: float = DEFAULT_WOMAC_WEIGHT) -> str:
    # 1) WOMAC z odpowiedzi użytkownika
    womac_score, max_score = calculate_womac_score(response)
    normalized = normalize_womac(womac_score, max_score)
    womac_label = womac_severity_label(normalized)                     # np. "Minimal functional limitation"

    # 2) Predykcja modelu tablicowego (KL-probabilities w kolejności MODEL_CLASSES)
    text_probs = predict_class(response)                               # np. [p(KL0), p(KL1), ..., p(KL4)]

    # 3) Połącz WOMAC z modelem (womac_weight domyślnie 0.7)
    combined_probs, class_list = combine_probs(womac_label, text_probs, womac_weight)

    # 4) Wybór klasy i mapowanie na etykietę do UI
    final_idx = int(np.argmax(combined_probs))
    final_kl = class_list[final_idx]                                   # np. "KL2"
    final_label = KL_TO_GRADE.get(final_kl, "Grade 2 - minimal")       # np. "Grade 2 - minimal"

    return final_label

GRADE_UI = {
    "Grade 0 - none": {
        "title": "No radiographic OA",
        "color": "#22c55e",  # green-500
        "icon": "check-circle",
        "advice": [
            "Keep up your activity routine.",
            "Maintain healthy weight and posture.",
            "If pain appears, try brief rest and gentle mobility."
        ],
    },
    "Grade 1 - doubtful": {
        "title": "Doubtful changes",
        "color": "#84cc16",  # lime-500
        "icon": "sparkles",
        "advice": [
            "Regular light activity (walking, cycling).",
            "Strengthen hips/quads 2–3×/week.",
            "Monitor pain trend over weeks."
        ],
    },
    "Grade 2 - minimal": {
        "title": "Minimal OA",
        "color": "#eab308",  # amber-500
        "icon": "circle-dot",
        "advice": [
            "Pace activities; alternate load and recovery.",
            "Targeted exercises for mobility + strength.",
            "Consider simple analgesics after consulting a clinician."
        ],
    },
    "Grade 3 - moderate": {
        "title": "Moderate OA",
        "color": "#f97316",  # orange-500
        "icon": "alert-triangle",
        "advice": [
            "Structured physio plan; low-impact cardio.",
            "Discuss pain management options with a clinician.",
            "Consider weight optimization if applicable."
        ],
    },
    "Grade 4 - severe": {
        "title": "Severe OA",
        "color": "#ef4444",  # red-500
        "icon": "octagon-alert",
        "advice": [
            "Consult an orthopedist/physiotherapist.",
            "Individualized pain strategy; assistive options.",
            "Discuss definitive options if daily function is limited."
        ],
    },
}

def build_result_card(responses: Dict, womac_weight: float = DEFAULT_WOMAC_WEIGHT) -> Dict:
    """
    Zwraca kompletny 'kafelek' dla UI:
    {
      grade_label, grade_title, color, icon,
      womac: {raw, max, percent, severity_label},
      model: {classes, probs, chosen},
      combined: {chosen_kl, chosen_grade},
      tips: [ ... ]
    }
    """
    # 1) WOMAC
    womac_raw, womac_max = calculate_womac_score(responses)
    womac_percent = round(normalize_womac(womac_raw, womac_max), 1)
    womac_sev = womac_severity_label(womac_percent)

    # 2) Model tablicowy
    text_probs = predict_class(responses)  # np. [p(KL0..KL4)]
    combined_probs, class_list = combine_probs(womac_sev, text_probs, womac_weight)
    final_idx = int(np.argmax(combined_probs))
    final_kl = class_list[final_idx]

    # 3) Mapowanie do etykiety UI
    # Jeśli używasz KL_TO_GRADE z wcześniejszej sekcji:
    grade_label = KL_TO_GRADE.get(final_kl, "Grade 2 - minimal")
    ui = GRADE_UI.get(grade_label, GRADE_UI["Grade 2 - minimal"])

    return {
        "grade_label": grade_label,         # np. "Grade 3 - moderate"
        "grade_title": ui["title"],         # zwięzły nagłówek
        "color": ui["color"],               # hex do paska/ikony
        "icon": ui["icon"],                 # nazwa ikony w Twoim frontcie

        "womac": {
            "raw": womac_raw,
            "max": womac_max,
            "percent": womac_percent,       # 0–100
            "severity_label": womac_sev,    # np. "Moderate functional limitation"
            "weight": womac_weight,         # ile WOMAC wnosi do fuzji
        },

        "model": {
            "classes": list(MODEL_CLASSES),              # np. ["KL0","KL1","KL2","KL3","KL4"]
            "probs": [float(x) for x in np.asarray(text_probs).ravel().tolist()],
            "chosen": final_kl,                           # np. "KL3"
        },

        "combined": {
            "probs": [float(x) for x in np.asarray(combined_probs).ravel().tolist()],
            "chosen_kl": final_kl,
            "chosen_grade": grade_label,
        },

        "tips": ui["advice"],              # krótkie wskazówki dopasowane do grade
    }
