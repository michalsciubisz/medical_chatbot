from utils.text_model import predict_class
from config.constant import WOMAC_CLASS_MAPPING, WOMAC_COLUMNS

import numpy as np

def womac_severity_level(normalized_score):
    if normalized_score <= 10:
        return "No functional issues"
    elif normalized_score <= 25:
        return "Mild functional limitation"
    elif normalized_score <= 45:
        return "Moderate functional limitation"
    elif normalized_score <= 65:
        return "Severe functional limitation"
    else:
        return "Extreme functional limitation"
    
def normalize_womac(score, max_score=96):
    return (score / max_score) * 100

def calculate_womac_score(responses):
    score = sum([int(responses[q]) for q in WOMAC_COLUMNS if q in responses])
    return score

def calculate_womac(response, womac_weight=0.7):

    womac_score = calculate_womac_score(response)

    normalized_score = normalize_womac(womac_score)
    womac_label = womac_severity_level(normalized_score)
    womac_class = WOMAC_CLASS_MAPPING[womac_label]

    womac_probs = np.zeros(5)
    womac_probs[womac_class] = 1.0

    text_probs = predict_class(response) 

    combined_probs = womac_weight * womac_probs + (1 - womac_weight) * text_probs

    combined_class_index = np.argmax(combined_probs)
    inverse_class_map = {v: k for k, v in WOMAC_CLASS_MAPPING.items()}
    combined_label = inverse_class_map[combined_class_index]

    return combined_label