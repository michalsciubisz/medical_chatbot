from __future__ import annotations
from typing import Any, Dict
from joblib import load
import pandas as pd

from config.constant import (
    COLUMNS_TO_FILL_PATIENT,
    MODEL_PATH,
    TEXT_DATA_CSV_PATH,
    MODEL_CLASSES_FALLBACK,
    ID_TO_LEGACY,
    LEGACY_TO_ID,
    WORK_TYPE_MAP,
    YN_MAP,
    DEFAULT_GENDER,
    DEFAULT_WORK_IF_NONE,
)

_MODEL = load(MODEL_PATH)
try:
    MODEL_CLASSES = list(getattr(_MODEL, "classes_", MODEL_CLASSES_FALLBACK))
except Exception:
    MODEL_CLASSES = list(MODEL_CLASSES_FALLBACK)

def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def rget(response: Dict[str, Any], key_or_id: str, default=None):
    if key_or_id in response:
        return response[key_or_id]

    legacy = ID_TO_LEGACY.get(key_or_id)
    if legacy and legacy in response:
        return response[legacy]

    if key_or_id in LEGACY_TO_ID:
        new_id = LEGACY_TO_ID[key_or_id]
        if new_id in response:
            return response[new_id]

    return default

def norm_work_type(val: Any) -> str:
    if val is None:
        return DEFAULT_WORK_IF_NONE
    v = str(val).strip().lower().replace(" ", "_")
    # spróbuj mapowania z podkreśleniami i bez
    if v in WORK_TYPE_MAP:
        return WORK_TYPE_MAP[v]
    v2 = v.replace("_", " ")
    return WORK_TYPE_MAP.get(v2, val)

def yn_token(x: Any, default: str = "nie") -> str:
    v = str(x).strip().lower() if x is not None else None
    return YN_MAP.get(v, default)

def create_df_from_response(response: Dict[str, Any]) -> pd.DataFrame:
    height = to_int(rget(response, "height_cm"))
    weight = to_int(rget(response, "weight_kg"))
    age = to_int(rget(response, "age"))
    work = norm_work_type(rget(response, "work_type"))

    bmi_calc = 0.0
    if height > 0:
        bmi_calc = weight / ((height / 100.0) ** 2)

    def s(id_or_legacy: str, default: int = 0) -> int:
        return to_int(rget(response, id_or_legacy), default)

    # Uwaga: NAZWY KOLUMN MUSZĄ EXACT MATCH do CSV/modelu.
    patient = {
        "Gender": DEFAULT_GENDER,
        "Age": age,
        "Weight": weight,
        "Height": height,
        "BMI": bmi_calc,
        "Work type": work,

        # Jeśli masz kolumny Diagnosis/Cause/Ailments → uzupełnij zerami:
        **{col: 0 for col in COLUMNS_TO_FILL_PATIENT
           if ("Diagnosis" in col) or ("Cause" in col) or ("Ailments" in col)},

        # Pola binarne/tekstowe – używaj tokenów jak w CSV
        "Family history with osteoarthisis": "nie",
        "Moving independently": "nie",
        "Use of assistive equipment": "nie",

        "Did you practice any sports or recreational activities before your illness": "recreational activities",
        "Forced to give up": "not affected",
        "Emotional disturbances": "not present",

        # Skale WOMAC (int 0–4) – dopasowane do nazw kolumn w CSV
        "Performing body toilet, bathing": s("phys_bath"),
        "Dressing": s("phys_put_socks"),
        "Cleaning": s("phys_light"),
        "Lifting or carrying heavy items": s("phys_heavy"),
        "Climbing stairs up to one floor": s("phys_ascend"),
        "Climbing stairs up to second floor": s("phys_ascend"),
        "Going down the stairs": s("phys_descend"),
        "Bending or crouching": s("phys_bending"),
        "Walk about 100 meters": s("phys_walk_flat"),
        "Walk about 500 meters": s("phys_walk_flat"),
        "Walk above 1000 meters": s("phys_walk_flat"),
        "Cycling": s("phys_cycling"),
        "Pursuing your own interests": s("pain_rest"),
        "How often did pain occur in last month": s("freq_last_month"),
        "Pain disturbed everyday life": s("disturb_daily"),

        "Does pain forced to use of painkillers": 0,
        "Does osteoarthritis affected your professional work": 0,
        "Therapy - pharmacological": 0.0,
        "Therapy - non-pharmacological": 0.0,
        "Therapy - surgical": 0.0,
        "Therapy - diet": 0.0,
    }

    return pd.DataFrame([patient])

def predict_class(response: Dict[str, Any]):
    """
    Zwraca wektor prawdopodobieństw (1D numpy array) w kolejności MODEL_CLASSES.
    """
    # 1) referencyjny DF by wyrównać one-hot
    df_ref = pd.read_csv(TEXT_DATA_CSV_PATH)
    # jeżeli te kolumny są w pliku – usuń jak dotąd
    for col in ("severity_score", "severity_grade"):
        if col in df_ref.columns:
            df_ref = df_ref.drop(columns=[col])

    # 2) DF pacjenta
    patient_df = create_df_from_response(response)

    # 3) One-hot + wyrównanie kolumn
    patient_hot = pd.get_dummies(patient_df)
    df_ref_hot = pd.get_dummies(df_ref)
    patient_hot = patient_hot.reindex(columns=df_ref_hot.columns, fill_value=0)

    # 4) Predykcja
    probs_2d = _MODEL.predict_proba(patient_hot)  # shape: (1, n_classes)
    return probs_2d[0]