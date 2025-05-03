from flask import Blueprint, request, jsonify
from utils.womac import calculate_womac
from utils.image_model import predict_image_classification, adjust_image
from config.constant import QUESTIONS, IMAGE_MAPPER

import uuid
import numpy as np

chatbot_bp = Blueprint("chatbot", __name__)

user_sessions = {}  #stored temporary in variable

@chatbot_bp.route("/api/start", methods=["POST"])
def start_conversation():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = {
        "responses": {},
        "current_question_index": 0,
        "mode": None  # will be either text or image
    }
    return jsonify({"session_id": session_id, "question": "Hello! Please choose type of classification you want to take! \n Please type 'image' or 'text' to choose classification type."})

@chatbot_bp.route("/api/respond", methods=["POST"])
def respond():
    data = request.json
    session_id = data["session_id"]
    answer = data.get("answer", "").strip()

    if not answer:
        answer = "0"

    user_data = user_sessions.get(session_id)

    womac_questions = QUESTIONS["womac"]
    current_index = user_data["current_question_index"]

    # handling first time user entry
    if current_index == 0 and user_data["mode"] is None:
        if answer.lower() == "image":
            user_data["mode"] = "image"
            return jsonify({"question": "Please upload an image now."})
        elif answer.lower() == "text":
            user_data["mode"] = "text"
            user_data["current_question_index"] += 1
            return jsonify({"question": womac_questions[0]})
        else:
            return jsonify({"question": "Please type 'image' or 'text' to choose classification type."})

    if user_data["mode"] == 'text':
        # saving previous answer
        if current_index - 1 < len(womac_questions):
            previous_question = womac_questions[current_index - 1]
            user_data["responses"][previous_question] = answer

        # checking if there sth more
        if current_index < len(womac_questions):
            next_question = womac_questions[current_index]
            user_data["current_question_index"] += 1
            return jsonify({"question": next_question})
        else:
            print(user_data['responses'])
            result = calculate_womac(user_data["responses"])
            return jsonify({
                "result": result,
                "message": "Thank you for completing the questionnaire."
            })

@chatbot_bp.route("/api/classify_image", methods=["POST"])
def classify_image():
    image = request.files["image"] # get image from request.files
    adjusted_image = adjust_image(image)
    prediction = predict_image_classification(adjusted_image)  # function to classify based on image
    predicted_class = np.argmax(prediction).item()  # because output is softmax
    return jsonify({"prediction": IMAGE_MAPPER[predicted_class]})

@chatbot_bp.route("/api/early_finish", methods=["POST"])
def early_finish():
    data = request.json
    session_id = data["session_id"]
    user_data = user_sessions.get(session_id)
    if not user_data:
        return jsonify({"error": "Invalid session"}), 400
    
    if user_data["mode"] == "text":

        responses = user_data["responses"]
        
        womac_questions = QUESTIONS["womac"][4:]  
        for question in womac_questions:
            if question not in responses:
                responses[question] = "0" 
        
        result = calculate_womac(responses)
        return jsonify({
            "result": f"Early assessment complete: {result}",
            "assessment": result,
            "is_partial": True,
            "message": "Note: Some answers were set to neutral (0) for calculation."
        })
    
    elif user_data["mode"] == "image":
        # Handle image case
        return jsonify({
            "result": "Image analysis complete",
            "message": "Thank you for your submission."
        })
    
    return jsonify({"error": "No assessment mode selected"}), 400