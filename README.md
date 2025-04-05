# 🩺 Chatbot Supporting the Classification of the Severity of Osteoarthritis

This project is a full-stack AI-powered chatbot system that assists users in identifying and classifying the severity of osteoarthritis using two input methods:
1. **Natural Language Input** – symptom descriptions typed by the user.
2. **Medical Imaging** – uploaded X-ray images of joints.

It leverages **NLP**, **CNN-based image classification**, and an intuitive **chatbot interface** to support users and potentially assist healthcare professionals.

---

## 🧠 Project Objectives

- Classify osteoarthritis severity based on:
  - Symptom descriptions (text)
  - X-ray image input
- Provide chatbot interaction for ease of use
- Enable an accessible web-based tool for preliminary osteoarthritis assessment

---

## 🧱 Technology Stack

### 🔙 Backend (Flask)
- Flask (REST API)
- Scikit-learn (NLP classifier)
- SentenceTransformers (for text embeddings)
- PyTorch (CNN image classification)
- Pillow, OpenCV (image preprocessing)

### 🔚 Frontend (React)
- React
- Basic chatbot UI components
- Image and text input forms

### 📦 Model Tools
- Pretrained NLP models (tbd.)
- Transfer learning with CNNs (tbd.)
- Jupyter for training workflows

---

## 🗂 Project Structure

```bash
medical_chatbot/
├── backend/
│   ├── app.py
│   ├── routes/
│   └── model/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   └── App.jsx
│   └── package.json
├── data/
│   ├── text_dataset.xlsx
│   └── images/
├── notebooks/
│   ├── train_nlp_model.ipynb
│   └── train_cnn_model.ipynb
└── README.md
```
## 🛠️ Installation and Setup

### Backend (Flask)
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

## 📄 License
This project is for academic purposes. For medical use, please consult professionals and validate the system rigorously.