from flask import Blueprint, request, jsonify, session
from utils.womac import calculate_womac
from utils.image_model import predict_image_classification, adjust_image
import uuid
import numpy as np

IMAGE_MAPPER = {
    0 : "healthy/doubtful",
    1 : "moderate",
    2 : "severe"
}

chatbot_bp = Blueprint("chatbot", __name__)

user_sessions = {}  #stored temporary in variable

@chatbot_bp.route("/api/start", methods=["POST"])
def start_conversation():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = {
        "responses": [],
        "current_question_index": 0,
        "mode": None  # will be either text or image
    }
    return jsonify({"session_id": session_id, "question": "Hello! Please choose type of classification you want to take! \n Please type 'image' or 'text' to choose classification type."})

@chatbot_bp.route("/api/respond", methods=["POST"])
def respond():
    data = request.json
    session_id = data["session_id"]
    answer = data["answer"]

    user_data = user_sessions.get(session_id)
    user_data["responses"].append(answer) # adding response to user_data dict

    questions = {
        'womac' : [
            "How old are you?",
            "Do you feel joint pain while walking?",
            "How often do you experience stiffness?",
            ]
    }

    # handling first time user entry
    if user_data["current_question_index"] == 0:
        if answer.lower() == "image":
            user_data["mode"] = "image"
            return jsonify({"question": "Please upload an image now."})
        elif answer.lower() == "text":
            user_data["mode"] = "text"
            user_data["current_question_index"] += 1
            return jsonify({"question": questions["womac"][0]})

    # going through the questions
    if user_data["current_question_index"] < len(questions["womac"]) and user_data["mode"] == 'text':
        question = questions["womac"][user_data["current_question_index"]]
        user_data["current_question_index"] += 1
        return jsonify({"question": question})
    
    # at the end calculate womac score
    if user_data["current_question_index"] >= len(questions["womac"]) and user_data["mode"] == 'text':
        result = calculate_womac(user_data["responses"])
        #TODO: implement classify_text here and connect two results to determine without no doubt
        return jsonify({"result": result})

@chatbot_bp.route("/api/classify_image", methods=["POST"])
def classify_image():
    image = request.files["image"] # get image from request.files
    adjusted_image = adjust_image(image)
    prediction = predict_image_classification(adjusted_image)  # function to classify based on image
    predicted_class = np.argmax(prediction).item()  # because output is softmax
    return jsonify({"prediction": IMAGE_MAPPER[predicted_class]})

# @chatbot_bp.route("/api/classify_text", methods=["POST"])
# def classify_text():
#     pass