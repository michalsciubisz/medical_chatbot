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
) -> Tuple[np.ndarray, List[str]]:
    text_probs = np.asarray(text_probs, dtype=float).ravel()
    if text_probs.size == 0:
        # Brak predykcji tekstowej -> użyj wyłącznie WOMAC (z kanonicznymi klasami)
        target_grade = LABEL_TO_GRADE[womac_label]
        womac_vec = np.zeros(len(GRADE_RANK), dtype=float)
        womac_vec[GRADE_RANK.index(target_grade)] = 1.0
        return womac_vec, list(GRADE_RANK)

    # Normalizacja rozkładu z modelu tekstowego
    s = text_probs.sum()
    if s > 0:
        text_probs = text_probs / s

    # Klasy z modelu tekstowego; mogą (ale nie muszą) być identyczne jak GRADE_RANK
    model_classes_local = _align_model_classes(MODEL_CLASSES)

    # Upewnij się, że długość klas = długość wektora prawdopodobieństw
    if len(model_classes_local) != len(text_probs):
        # Jeśli rozjazd, przytnij/rozszerz w bezpieczny sposób:
        k = min(len(model_classes_local), len(text_probs))
        model_classes_local = model_classes_local[:k]
        text_probs = text_probs[:k]

    # Wyznacz klasę WOMAC w przestrzeni dostępnych klas modelu
    target_grade = LABEL_TO_GRADE[womac_label]
    mapped_grade = (
        target_grade if target_grade in model_classes_local
        else _closest_grade(target_grade, model_classes_local)
    )

    womac_vec = np.zeros(len(model_classes_local), dtype=float)
    womac_vec[model_classes_local.index(mapped_grade)] = 1.0

    combined = womac_weight * womac_vec + (1.0 - womac_weight) * text_probs
    return combined, model_classes_local

def calculate_womac(response: Dict, womac_weight: float = DEFAULT_WOMAC_WEIGHT) -> str:
    """
    Główna funkcja: oblicz wynik WOMAC, znormalizuj do %, przypisz etykietę,
    połącz z predykcją modelu tekstowego i zwróć końcową etykietę (string).
    """
    womac_score, max_score = calculate_womac_score(response)
    normalized = normalize_womac(womac_score, max_score)
    womac_label = womac_severity_label(normalized)

    text_probs = predict_class(response)  # spodziewamy się wektora o długości == len(MODEL_CLASSES)
    combined_probs, class_list = combine_probs(womac_label, text_probs, womac_weight)

    idx = int(np.argmax(combined_probs))
    combined_grade = class_list[idx]

    # Spróbuj zmapować na WOMAC label; jeśli nieznana klasa, fallback do WOMAC z ankiety
    combined_label = GRADE_TO_LABEL.get(combined_grade, womac_label)

    # Debug (opcjonalnie — możesz podpiąć pod logger)
    # print(f"WOMAC raw: {womac_score}/{max_score} -> {normalized:.2f}% -> {womac_label}")
    # print(f"Model classes used: {class_list} (len={len(class_list)})")
    # print(f"Text probs: {np.asarray(text_probs).ravel()}")
    # print(f"Combined:  {combined_probs} -> {combined_grade} -> {combined_label}")

    return combined_label
