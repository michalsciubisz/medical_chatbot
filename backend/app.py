from flask import Flask
from flask_cors import CORS
from routes.chatbot import chatbot_bp

app = Flask(__name__)
CORS(app)  

app.register_blueprint(chatbot_bp)

if __name__ == "__main__":
    app.run(debug=True)