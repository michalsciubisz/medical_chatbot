# utils/text_model.py
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# === ŚCIEŻKI – dostosuj do siebie ===
MODEL_PATH = r"D:/semestr_10/master_thesis/medical_chatbot/artifacts/best_model.joblib"
# opcjonalnie: jeśli masz ten plik z treningu – przydaje się do debugów/odwzorowania nazw
FEATURES_PATH = r"D:/semestr_10/master_thesis/medical_chatbot/artifacts/feature_columns.json"

# Twoje klasy z treningu będą typu: ["KL0","KL1","KL2","KL3","KL4"] (kolejność wg model.classes_)
# Wczytujemy model jednokrotnie:
_model = joblib.load(MODEL_PATH)
MODEL_CLASSES: List[str] = list(_model.classes_)

# Mapa do etykiet „Grade …” (jeśli używasz ich gdzieś w UI)
KL_TO_GRADE = {
    "KL0": "Grade 0 - none",
    "KL1": "Grade 1 - doubtful",
    "KL2": "Grade 2 - minimal",
    "KL3": "Grade 3 - moderate",
    "KL4": "Grade 4 - severe",
}
GRADE_TO_KL = {v: k for k, v in KL_TO_GRADE.items()}

# === MAPOWANIE ODPOWIEDZI -> SUROWE KOLUMNY CSV ===
# Zbieramy tylko to, co masz już w pytaniach lub łatwo dodać.
# Reszta może być NaN – pipeline ma imputer.
# Nazwy kluczy w 'responses' to Twoje id (np. age, height_cm, weight_kg, pain_walk, ...)

def _bmi_from(height_cm: float, weight_kg: float) -> float:
    try:
        h = float(height_cm) / 100.0
        w = float(weight_kg)
        if h > 0:
            return w / (h * h)
    except Exception:
        pass
    return np.nan

def _to_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _to_int01(x):
    # do binarnych 0/1 (np. pain flags)
    try:
        v = int(float(x))
        return 1 if v > 0 else 0
    except Exception:
        return 0

# Najczęstsze, wpływowe kolumny, które mamy/łatwo dodać:
# - demografia: sex, age_at_the_start_of_the_study, body_mass_index
# - WOMAC składowe (już pytasz o skale 0-4) -> możemy zasilić nimi pola pokrewne
# - surowe symptomy: pain_* (z Twoich pytań)
#
# UWAGA: nazwy po PRAWEJ to *dokładnie* kolumny w Twoim CSV (te z nagłówka).
# Po LEWEJ – ID z Twoich pytań / odpowiedzi.
#
# Jeśli jakiejś kolumny nie dostarczysz → zostaje NaN → imputer zrobi robotę.
def responses_to_raw_row(responses: Dict) -> Dict:
    age = _to_num(responses.get("age"))
    height = _to_num(responses.get("height_cm"))
    weight = _to_num(responses.get("weight_kg"))
    bmi = _bmi_from(height, weight)

    # sex: 1=male, 2=female w Twoim CSV (zgodnie z wymową OA datasetów)
    # Jeżeli w UI nie masz pytania o płeć – ustaw stały default (np. 2 = kobieta)
    sex_map = {"man": 1, "male": 1, "m": 1, "kobieta": 2, "woman": 2, "female": 2, "k": 2}
    sex_val = responses.get("sex", "woman")
    sex = sex_map.get(str(sex_val).lower(), 2)

    # subscale* to skale 0..100 w CSV – my mamy 0..4; przeskalujemy *z grubsza* x25
    def s4_to_100(qid: str) -> float:
        v = _to_num(responses.get(qid))
        if np.isnan(v):
            return np.nan
        return max(0.0, min(4.0, v)) * 25.0

    # najbliższe odpowiedniki – nie są idealnie 1:1, ale pipeline i tak uśredni
    subscale_pain = s4_to_100("pain_walk")  # przybliżenie
    subscale_stiff = s4_to_100("stiff_morning")
    subscale_pf = s4_to_100("rising_sit")  # physical functioning -> wybrana jedna skala jako proxy

    # WOMAC TOTAL z Twoich pytań (jeśli będziesz potrzebował – na razie nie podajemy bezpośrednio)
    # Możesz też dodać inne „proxy” dla kolumn 0/1: pain_left_knee itp., ale nie musisz.

    row = {
        # — demografia i baza
        "sex": sex,
        "age_at_the_start_of_the_study": age,
        "body_mass_index": bmi,

        # — kilka subskal SF-36 (proxy z 0-4 → 0-100)
        "subscale_bodily_pain_sf36": subscale_pain,
        "subscale_vitality_energy_fatigue_sf36": s4_to_100("stiff_day"),
        "subscale_physical_functioning_sf36": subscale_pf,

        # — NRS bólu (proxy z wybranych pytań)
        "numeric_rating_scale_nrs_for_pain_intensity_of_hip_and__or_knee__right_now": _to_num(responses.get("pain_rest")),
        "nrs_for_pain_intensity_of_hip_and__or_knee__the_past_week": _to_num(responses.get("pain_night")),
        # Reszta pól zostaje NaN – to OK.
    }
    return row

def _expected_raw_columns_and_types():
    """
    Zwraca (raw_cols, col_type_map), gdzie:
      - raw_cols: lista kolumn surowych użytych w treningu (dokładnie te, których oczekuje ColumnTransformer)
      - col_type_map: dict {col_name: 'num'|'cat'} do późniejszego rzutowania typów
    """
    pre = _model.named_steps.get("pre")
    if pre is None:
        # model bez preprocesora – rzadko, ale obsłużmy
        return [], {}

    raw_cols = []
    col_type_map = {}
    for name, transformer, cols in pre.transformers_:
        if cols is None or cols == "drop":
            continue
        cols = list(cols)
        raw_cols.extend(cols)
        kind = "num" if name == "num" else ("cat" if name == "cat" else "num")
        for c in cols:
            col_type_map[c] = kind
    return raw_cols, col_type_map


def _align_to_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindeksuje wejściowy DF do surowych kolumn z treningu i ustawia odpowiednie typy.
    Brakujące kolumny -> NaN (imputer w preprocesorze sobie poradzi).
    """
    raw_cols, col_type_map = _expected_raw_columns_and_types()

    # Jeśli z jakiegoś powodu nie znaleźliśmy preprocesora – zwróć df bez zmian
    if not raw_cols:
        return df

    # Reindeks do oczekiwanych kolumn
    out = df.reindex(columns=raw_cols)

    # Ustaw typy – ważne dla OneHot/Imputera
    for c, kind in col_type_map.items():
        if kind == "cat":
            out[c] = out[c].astype("object")
        else:
            # numeryczne – trzymaj jako float (imputer='median' tego oczekuje)
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _to_dataframe_one(responses: Dict) -> pd.DataFrame:
    row = responses_to_raw_row(responses)
    df = pd.DataFrame([row])
    df = _align_to_training_columns(df)  # <--- KLUCZOWA LINIA
    return df

def predict_class(responses: Dict) -> np.ndarray:
    """
    Zwraca wektor P(class) w kolejności MODEL_CLASSES (np. ['KL0','KL1','KL2','KL3','KL4']).
    """
    X = _to_dataframe_one(responses)
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        # shape: (1, n_classes)
        return proba[0]
    # fallback (nie powinno zajść dla GradientBoostingClassifier)
    preds = _model.predict(X)
    onehot = np.zeros(len(MODEL_CLASSES), dtype=float)
    cls = str(preds[0])
    if cls in MODEL_CLASSES:
        onehot[MODEL_CLASSES.index(cls)] = 1.0
    return onehot
