import React, { useState, useEffect } from "react";
import axios from "axios";

function Chatbot() {
    const [sessionId, setSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [image, setImage] = useState(null); // storing the image
    const [result, setResult] = useState(null);

    useEffect(() => {
        axios.post("http://localhost:5000/api/start").then(res => {
            setSessionId(res.data.session_id);
            setMessages([{ text: res.data.question, sender: "bot" }]);
        });
    }, []);

    const handleImageUpload = (e) => {
        setImage(e.target.files[0]);
    };

    const submitImage = async () => {
        if (!image) return;
        const formData = new FormData();
        formData.append('image', image);
        try {
            const response = await axios.post("http://localhost:5000/api/classify_image", formData, {
                headers: { "Content-Type": "multipart/form-data" }
            });
            setResult(response.data.prediction);
        } catch (error) {
            console.error("Image classification failed:", error);
            setResult("Error: Could not classify image.");
        }
    };

    const handleSend = async () => {
        setMessages([...messages, { text: input, sender: "user" }]);
        const res = await axios.post("http://localhost:5000/api/respond", {
            session_id: sessionId,
            answer: input
        });
        setMessages(prev => [...prev, { text: res.data.question || res.data.result, sender: "bot" }]);
        setInput("");
    };

    return (
        <div>
            <div>
                {messages.map((m, i) => (
                    <div key={i} className={m.sender === "bot" ? "bot" : "user"}>
                        {m.text}
                    </div>
                ))}
            </div>
            <input value={input} onChange={e => setInput(e.target.value)} />
            <button onClick={handleSend}>Send</button>

            <hr />
            
            <div style={{ marginTop: "20px", border: "1px solid #ccc", padding: "10px" }}>
                <h3>Image Classification</h3>
                <input type="file" accept="image/*" onChange={handleImageUpload} />
                <button onClick={submitImage} disabled={!image}>Classify Image</button>

                {result && (
                    <div style={{ marginTop: "10px" }}>
                        <strong>Your condition is predicted to be:</strong> {result}
                    </div>
                )}
            </div>
        </div>
    );
}

export default Chatbot;
