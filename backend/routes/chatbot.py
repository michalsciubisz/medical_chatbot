from flask import Blueprint, request, jsonify
from config.constant import QUESTIONS, IMAGE_MAPPER, MIN_ANSWERS, IMAGE_TO_KL
from utils.womac import calculate_womac, build_result_card
from utils.image_model import predict_image_classification, adjust_image, build_image_result_card
from utils.json_sanitize import _json_sanitize
import uuid, numpy as np
from utils.text_model import predict_class, MODEL_CLASSES

KL_TO_GRADE = {
    "KL0": "Grade 0 - none",
    "KL1": "Grade 1 - doubtful",
    "KL2": "Grade 2 - minimal",
    "KL3": "Grade 3 - moderate",
    "KL4": "Grade 4 - severe",
}
GRADE_TO_KL = {v: k for k, v in KL_TO_GRADE.items()}

chatbot_bp = Blueprint("chatbot", __name__)
user_sessions = {}



def validate_answer(qdef, answer):
    t = qdef["type"]
    if t == "number":
        try:
            val = float(answer)
        except:
            return False, "Please enter a valid number."
        if "min" in qdef and val < qdef["min"]:
            return False, f"Value must be ≥ {qdef['min']}."
        if "max" in qdef and val > qdef["max"]:
            return False, f"Value must be ≤ {qdef['max']}."
        return True, val
    elif t == "choice":
        if answer not in qdef["choices"]:
            return False, f"Choose one of: {', '.join(qdef['choices'])}."
        return True, answer
    elif t == "scale":
        try:
            val = int(answer)
        except:
            return False, "Please choose a value on the scale."
        lo, hi = qdef.get("min", 0), qdef.get("max", 4)
        if val < lo or val > hi:
            return False, f"Value must be between {lo} and {hi}."
        return True, val
    else:
        # fallback na tekst
        return True, answer

@chatbot_bp.route("/api/start", methods=["POST"])
def start_conversation():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = {
        "responses": {},
        "current_question_index": 0,
        "mode": None
    }
    return jsonify({
        "session_id": session_id,
        "question": "Hello! Please choose type of assessment: type 'image' or 'text'."
    })

# @chatbot_bp.route("/api/respond", methods=["POST"])
# def respond():
#     data = request.json or {}
#     session_id = data.get("session_id")
#     answer = (data.get("answer") or "").strip()

#     if not session_id or session_id not in user_sessions:
#         return jsonify({"error":"Invalid or expired session."}), 400

#     s = user_sessions[session_id]
#     womac = QUESTIONS["womac"]
#     idx = s["current_question_index"]

#     # wybór trybu
#     if idx == 0 and s["mode"] is None:
#         if answer.lower() == "image":
#             s["mode"] = "image"
#             return jsonify({"question":"Please upload an image now.", "mode":"image"})
#         elif answer.lower() == "text":
#             s["mode"] = "text"
#             # nie pobieramy odpowiedzi, odpalamy 1. pytanie
#             q = womac[0]
#             s["current_question_index"] = 1
#             return jsonify({"question": q["text"], "question_id": q["id"], "type": q["type"], "meta": {k:q[k] for k in ("min","max","choices","labels") if k in q}})
#         else:
#             return jsonify({"question":"Please type 'image' or 'text' to choose classification type."})

#     if s["mode"] != "text":
#         return jsonify({"error":"Text flow not active."}), 400

#     # zapis i walidacja odpowiedzi do poprzedniego pytania
#     prev_idx = idx - 1
#     if prev_idx >= 0 and prev_idx < len(womac):
#         prev_q = womac[prev_idx]
#         ok, val_or_msg = validate_answer(prev_q, answer if answer != "" else "0")
#         if not ok:
#             # zwróć ten sam prompt + komunikat
#             return jsonify({
#                 "error":"validation_error",
#                 "message": val_or_msg,
#                 "question": prev_q["text"],
#                 "question_id": prev_q["id"],
#                 "type": prev_q["type"],
#                 "meta": {k:prev_q[k] for k in ("min","max","choices","labels") if k in prev_q}
#             }), 400
#         s["responses"][prev_q["id"]] = val_or_msg

#     # kolejne pytanie lub wynik
#     if idx < len(womac):
#         q = womac[idx]
#         s["current_question_index"] += 1
#         return jsonify({
#             "question": q["text"],
#             "question_id": q["id"],
#             "type": q["type"],
#             "meta": {k:q[k] for k in ("min","max","choices","labels") if k in q}
#         })
#     else:
#         result = calculate_womac(s["responses"])
#         return jsonify({"result": result, "message":"Thank you for completing the questionnaire."})

@chatbot_bp.route("/api/respond", methods=["POST"])
def respond():
    data = request.json or {}
    session_id = data.get("session_id")
    answer = (data.get("answer") or "").strip()

    if not session_id or session_id not in user_sessions:
        return jsonify({"error":"Invalid or expired session."}), 400

    s = user_sessions[session_id]
    womac = QUESTIONS["womac"]
    idx = s["current_question_index"]

    # wybór trybu
    if idx == 0 and s["mode"] is None:
        if answer.lower() == "image":
            s["mode"] = "image"
            return jsonify({"question":"Please upload an image now.", "mode":"image"})
        elif answer.lower() == "text":
            s["mode"] = "text"
            q = womac[0]
            s["current_question_index"] = 1
            return jsonify({"question": q["text"], "question_id": q["id"], "type": q["type"], "meta": {k:q[k] for k in ("min","max","choices","labels") if k in q}})
        else:
            return jsonify({"question":"Please type 'image' or 'text' to choose classification type."})

    if s["mode"] != "text":
        return jsonify({"error":"Text flow not active."}), 400

    # zapis i walidacja odpowiedzi do poprzedniego pytania
    prev_idx = idx - 1
    if prev_idx >= 0 and prev_idx < len(womac):
        prev_q = womac[prev_idx]
        ok, val_or_msg = validate_answer(prev_q, answer if answer != "" else "0")
        if not ok:
            return jsonify({
                "error":"validation_error",
                "message": val_or_msg,
                "question": prev_q["text"],
                "question_id": prev_q["id"],
                "type": prev_q["type"],
                "meta": {k:prev_q[k] for k in ("min","max","choices","labels") if k in prev_q}
            }), 400
        s["responses"][prev_q["id"]] = val_or_msg

    # kolejne pytanie lub wynik
    if idx < len(womac):
        q = womac[idx]
        s["current_question_index"] += 1
        return jsonify({
            "question": q["text"],
            "question_id": q["id"],
            "type": q["type"],
            "meta": {k:q[k] for k in ("min","max","choices","labels") if k in q}
        })
    else:
        card = build_result_card(s["responses"])
        safe_card = _json_sanitize(card)
        # wsteczna zgodność: stary front nadal odczyta sam string
        result_text = safe_card.get("grade_label", "Unknown")

        return jsonify({
            "result": result_text,         # ← string jak dawniej
            "card": safe_card,             # ← nowy, bogaty obiekt
            "message": "Thank you for completing the questionnaire."
        })

# @chatbot_bp.route("/api/classify_image", methods=["POST"])
# def classify_image():
#     image = request.files["image"] # get image from request.files
#     adjusted_image = adjust_image(image)
#     prediction = predict_image_classification(adjusted_image)  # function to classify based on image
#     predicted_class = np.argmax(prediction).item()  # because output is softmax
#     return jsonify({"prediction": IMAGE_MAPPER[predicted_class]})

# @chatbot_bp.route("/api/classify_image", methods=["POST"])
# def classify_image():
#     if "image" not in request.files:
#         return jsonify({"error": "No image file provided under field 'image'."}), 400

#     file = request.files["image"]

#     try:
#         adjusted_image = adjust_image(file)
#         prediction = predict_image_classification(adjusted_image)  # shape (1, C)
#         probs = prediction[0]
        
#         # sanity-check: liczba klas = długość IMAGE_MAPPER
#         C = len(probs)
#         if C != len(IMAGE_MAPPER):
#             return jsonify({"error": f"Model returned {C} classes, but IMAGE_MAPPER has {len(IMAGE_MAPPER)}."}), 500

#         pred_idx = int(np.argmax(probs))
#         pred_label = IMAGE_MAPPER[pred_idx]                 # "moderate"
#         pred_kl = IMAGE_TO_KL.get(pred_label)               # np. "KL3"
#         confidence = float(probs[pred_idx])

#         # top-N w KL
#         top_n = min(3, C)
#         top_indices = np.argsort(probs)[::-1][:top_n]
#         top_list = []
#         for i in top_indices:
#             i = int(i)
#             lbl = IMAGE_MAPPER[i]                           # "moderate"
#             kl = IMAGE_TO_KL.get(lbl, "KL3")                # domyśl np. KL3
#             top_list.append({"kl": kl, "prob": float(probs[i])})

#         # słownik {KLx: p}
#         probs_dict = {}
#         for i in range(C):
#             lbl = IMAGE_MAPPER[i]
#             kl = IMAGE_TO_KL.get(lbl, None)
#             if kl:
#                 probs_dict[kl] = float(probs[i])

#         # zbuduj kartę
#         card = build_image_result_card(
#             kl_code=pred_kl or "KL3",                       # fallback, gdyby mapping nie trafił
#             confidence=confidence,
#             probs=probs_dict
#         )
#         safe_card = _json_sanitize(card)

#         return jsonify({
#             "prediction": pred_kl or pred_label,            # zwróć KL jeśli jest, inaczej surową etykietę
#             "confidence": confidence,
#             "top": top_list,
#             "card": safe_card,
#             "message": "Image classification complete."
#         })
#     except Exception as e:
#         # warto logować pełny traceback w serwerze; do klienta krótki komunikat
#         return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

@chatbot_bp.route("/api/classify_image", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field 'image'."}), 400

    file = request.files["image"]
    try:
        adjusted = adjust_image(file)
        pred = predict_image_classification(adjusted)   # (1, C)
        probs = pred[0]
        C = len(probs)

        # sanity-check: czy rozmiar wyjścia pasuje do IMAGE_MAPPER?
        if C != len(IMAGE_MAPPER):
            return jsonify({
                "error": f"Model returned {C} classes, but IMAGE_MAPPER has {len(IMAGE_MAPPER)}."
            }), 500

        # najlepsza klasa
        pred_idx = int(np.argmax(probs))
        raw_label = IMAGE_MAPPER[pred_idx]               # "moderate"
        pred_kl = IMAGE_TO_KL.get(raw_label)             # "KL3"
        confidence = float(probs[pred_idx])

        # TOP-N
        top_n = min(3, C)
        top_indices = np.argsort(probs)[::-1][:top_n]
        top_list = []
        for i in top_indices:
            i = int(i)
            lbl = IMAGE_MAPPER[i]                        # "healthy/doubtful" | "moderate" | "severe"
            kl = IMAGE_TO_KL.get(lbl, None)              # "KL1"/"KL3"/"KL4"
            top_list.append({
                "label": lbl,
                "kl": kl or "N/A",
                "prob": float(probs[i]),
            })

        # rozkład w KL (tylko te, które mamy mapowalne)
        probs_dict = {}
        for i in range(C):
            lbl = IMAGE_MAPPER[i]
            kl = IMAGE_TO_KL.get(lbl, None)
            if kl:
                probs_dict[kl] = float(probs[i])

        # budowa karty
        card = build_image_result_card(
            kl_code=pred_kl or "KL3",
            confidence=confidence,
            probs=probs_dict
        )
        safe_card = _json_sanitize(card)

        return jsonify({
            # na potrzeby zgodności: zwracamy KL (spójne z tekstem)
            "prediction": pred_kl or raw_label,
            "confidence": confidence,
            "top": top_list,
            "card": safe_card,
            "message": "Image classification complete."
        })
    except Exception as e:
        # do logów serwera dorzuć traceback; klientowi krótki opis
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

# @chatbot_bp.route("/api/early_finish", methods=["POST"])
# def early_finish():
#     data = request.json or {}
#     session_id = data.get("session_id")
#     s = user_sessions.get(session_id)
#     if not s: return jsonify({"error":"Invalid session"}), 400
#     if s["mode"] != "text":
#         return jsonify({"result":"Image analysis complete","message":"Thank you for your submission."})

#     answered = len(s["responses"])
#     if answered < MIN_ANSWERS:
#         return jsonify({
#             "result": None,
#             "message": f"You've answered only {answered} questions. Continue or finish with low confidence?",
#             "low_confidence": True
#         })

#     # uzupełnij brakujące skale zerami tylko dla pytań typu "scale"
#     womac = QUESTIONS["womac"]
#     for q in womac:
#         if q["type"] == "scale" and q["id"] not in s["responses"]:
#             s["responses"][q["id"]] = 0

#     # result = calculate_womac(s["responses"])
#     card = build_result_card(s["responses"])
#     safe_card = _json_sanitize(card)
#     return jsonify({
#         "result": safe_card.get("grade_label", "Unknown"),
#         "card": safe_card,
#         "is_partial": True,
#         "message": "Note: Missing scale answers were set to 0 for calculation."
#     })

@chatbot_bp.route("/api/early_finish", methods=["POST"])
def early_finish():
    data = request.json or {}
    session_id = data.get("session_id")
    s = user_sessions.get(session_id)
    if not s:
        return jsonify({"error":"Invalid session"}), 400

    # tryb tekstowy wymagany
    if s["mode"] != "text":
        return jsonify({
            "result": "Image analysis complete",
            "message": "Thank you for your submission."
        })

    answered = len(s["responses"])
    if answered < MIN_ANSWERS:
        return jsonify({
            "result": None,
            "message": f"You've answered only {answered} questions. Continue or finish with low confidence?",
            "low_confidence": True
        })

    # wypełnij brakujące pytania typu "scale" zerami, żeby dało się policzyć WOMAC
    womac = QUESTIONS["womac"]
    for q in womac:
        if q["type"] == "scale" and q["id"] not in s["responses"]:
            s["responses"][q["id"]] = 0

    # zamiast surowego tekstu → zbuduj kartę
    card = build_result_card(s["responses"])
    label = card.get("grade_label", "Unknown")

    return jsonify({
        "result": f"Early assessment complete: {label}",
        "assessment": label,
        "is_partial": True,
        "message": "Note: Missing scale answers were set to 0 for calculation.",
        "card": card,
        # opcjonalnie, żeby od razu zasilić badge ryzyka:
        "text_risk": card.get("risk_label")
    })
