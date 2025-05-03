import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./Chatbot.css";

function Chatbot() {
    const [sessionId, setSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [image, setImage] = useState(null);
    const [result, setResult] = useState(null);
    const [activeTab, setActiveTab] = useState("chat");
    const [isExamComplete, setIsExamComplete] = useState(false);
    const [earlyFinishWarning, setEarlyFinishWarning] = useState(false);
    const [imageModeActive, setImageModeActive] = useState(false);
    const messagesEndRef = useRef(null);

    // scale configuration
    const scaleLabels = ["None", "Slight", "Moderate", "Very", "Extremely"];

    // scroll to chatbot
    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    // initialize converstaion
    useEffect(() => {
        axios.post("http://localhost:5000/api/start").then(res => {
            setSessionId(res.data.session_id);
            setMessages([{ 
                text: "Hello! Please choose type of assessment:\nType 'text' for questionnaire or 'image' for MRI analysis", 
                sender: "bot" 
            }]);
        });
    }, []);

    // scale questions
    const isScaleQuestion = (text) => {
        return text.includes("on scale 0-4");
    };

    // rendering questions with apriopriate question
    const renderQuestion = (text, sender) => {
        if (sender === 'bot' && isScaleQuestion(text)) {
            const questionText = text.split("\n")[1];
            return (
                <div className="scale-question">
                    <p>{questionText}</p>
                    <div className="rating-scale">
                        {scaleLabels.map((label, index) => (
                            <button 
                                key={index} 
                                onClick={() => handleScaleResponse(index)}
                                className="scale-option"
                            >
                                <span className="number">{index}</span>
                                <span className="label">{label}</span>
                            </button>
                        ))}
                    </div>
                </div>
            );
        }
        return <p>{text}</p>;
    };

    // hadnling scale response
    const handleScaleResponse = (value) => {
        setInput(value.toString());
        handleSend();
    };

    // restarting conversation
    const restartConversation = async () => {
        try {
            const res = await axios.post("http://localhost:5000/api/start");
            setSessionId(res.data.session_id);
            setMessages([{ 
                text: "Hello! Please choose type of assessment:\nType 'text' for questionnaire or 'image' for MRI analysis", 
                sender: "bot" 
            }]);
            setResult(null);
            setImage(null);
            setIsExamComplete(false);
            setEarlyFinishWarning(false);
            setImageModeActive(false);
        } catch (error) {
            console.error("Error restarting conversation:", error);
        }
    };

    // handling early finish
    const handleEarlyFinish = async () => {
        try {
            const res = await axios.post("http://localhost:5000/api/early_finish", {
                session_id: sessionId
            });
            
            if (res.data.result) {
                setMessages(prev => [
                    ...prev, 
                    { text: res.data.result, sender: "bot" },
                    { text: res.data.message, sender: "bot" }
                ]);
                setIsExamComplete(true);
            }
        } catch (error) {
            console.error("Error finishing early:", error);
            setMessages(prev => [...prev, { 
                text: "Error completing assessment early. Please try again.", 
                sender: "bot" 
            }]);
        }
    };

    // uploading image
    const handleImageUpload = (e) => {
        setImage(e.target.files[0]);
    };

    // submiting message
    const submitImage = async () => {
        if (!image) return;
        const formData = new FormData();
        formData.append('image', image);
        try {
            const response = await axios.post("http://localhost:5000/api/classify_image", formData, {
                headers: { "Content-Type": "multipart/form-data" }
            });
            setResult(response.data.prediction);
            setMessages(prev => [...prev, { 
                text: `Based on MRI imaging you are likely to have: ${response.data.prediction}.`, 
                sender: "bot" 
            }]);
        } catch (error) {
            console.error("Image classification failed:", error);
            setResult("Error: Could not classify image.");
        }
    };

    // handling sending message
    const handleSend = async () => {
        const trimmedInput = input.trim();
        if (!trimmedInput) return;
        
        // handling special commands
        if (trimmedInput.toLowerCase() === '/restart') {
            await restartConversation();
            setInput("");
            return;
        }
        
        if (trimmedInput.toLowerCase() === '/finish') {
            await handleEarlyFinish();
            setInput("");
            return;
        }

        // handling mode selection
        if (messages.length === 1 && (trimmedInput.toLowerCase() === 'image' || trimmedInput.toLowerCase() === 'text')) {
            if (trimmedInput.toLowerCase() === 'image') {
                setImageModeActive(true);
                setMessages(prev => [...prev, { 
                    text: "Please upload your MRI image using the upload section below.", 
                    sender: "bot" 
                }]);
                setInput("");
                return;
            }
        }
        
        // normal conversation
        const userMessage = { text: trimmedInput, sender: "user" };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        
        try {
            const res = await axios.post("http://localhost:5000/api/respond", {
                session_id: sessionId,
                answer: trimmedInput
            });
            
            if (res.data.question || res.data.result) {
                setMessages(prev => [...prev, { 
                    text: res.data.question || res.data.result, 
                    sender: "bot" 
                }]);
                
                if (res.data.result) {
                    setIsExamComplete(true);
                }
            }
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages(prev => [...prev, { 
                text: "Sorry, I encountered an error. Please try again.", 
                sender: "bot" 
            }]);
        }
    };

    return (
        <div className="chatbot-container">
            <header className="chatbot-header">
                <div className="header-content">
                    <img 
                        src="logo.png" 
                        alt="Medical Logo" 
                        className="header-logo"
                    />
                    <div>
                        <h1>chatOA</h1>
                        <p>chatbot osteoarthritis interpretation</p>
                    </div>
                </div>
                <p className="subtitle">Check if you have osteoarthritis!</p>
            </header>

            <div className="chatbot-tabs">
                <button 
                    className={activeTab === "chat" ? "active" : ""}
                    onClick={() => setActiveTab("chat")}
                >
                    Chat
                </button>
                <button 
                    className={activeTab === "results" ? "active" : ""}
                    onClick={() => setActiveTab("results")}
                >
                    Results
                </button>
                <button 
                    className={activeTab === "info" ? "active" : ""}
                    onClick={() => setActiveTab("info")}
                >
                    About OA
                </button>
            </div>

            {activeTab === "chat" && (
                <div className="chat-section">
                    <div className="command-hints">
                        <p>Type <strong>/restart</strong> to start over</p>
                        <p>Type <strong>/finish</strong> to complete early</p>
                    </div>

                    {earlyFinishWarning && (
                        <div className="warning-message">
                            <p>You haven't answered enough questions for a reliable assessment.</p>
                            <div>
                                <button onClick={() => setEarlyFinishWarning(false)}>Continue</button>
                                <button onClick={handleEarlyFinish}>Finish Anyway</button>
                            </div>
                        </div>
                    )}


                    <div className="messages-container">
                        {messages.map((m, i) => (
                            <div key={i} className={`message ${m.sender}`}>
                                {renderQuestion(m.text, m.sender)}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="input-area">
                        <input 
                            value={input} 
                            onChange={e => setInput(e.target.value)} 
                            placeholder="Type your message here..."
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        />
                        <button onClick={handleSend}>Send</button>
                    </div>

                    {imageModeActive && (
                        <div className="image-upload-section">
                            <h3>Provide your MRI scan:</h3>
                            <div className="upload-box">
                                {image ? (
                                    <div className="upload-preview">
                                        <span>{image.name}</span>
                                        <button onClick={() => setImage(null)}>Remove</button>
                                    </div>
                                ) : (
                                    <>
                                        <p>Upload image here:</p>
                                        <input 
                                            type="file" 
                                            accept="image/*" 
                                            onChange={handleImageUpload} 
                                            id="image-upload"
                                        />
                                        <label htmlFor="image-upload">Choose File</label>
                                    </>
                                )}
                            </div>
                            {image && (
                                <button 
                                    className="classify-button" 
                                    onClick={submitImage}
                                >
                                    Classify image
                                </button>
                            )}
                        </div>
                    )}
                </div>
            )}

            {activeTab === "results" && (
                <div className="results-section">
                    <h2>Summary of results:</h2>
                    <div className="result-card">
                        <h3>Based on text interview</h3>
                        <p>You are likely to have: {messages.length > 2 ? "some symptoms of OA" : "not enough data"}</p>
                    </div>
                    <div className="result-card">
                        <h3>Based on MRI imaging</h3>
                        <p>{result ? `You are likely to have: ${result}` : "No image analyzed yet"}</p>
                    </div>
                </div>
            )}

            {activeTab === "info" && (
                <div className="info-section">
                    <h2>About Osteoarthritis</h2>
                    <p>Osteoarthritis (OA) is the most common form of arthritis, affecting millions of people worldwide.</p>
                    
                    <h3>How does this chatbot work?</h3>
                    <p>The chatbot analyzes your symptoms through conversation and can evaluate MRI images.</p>

                    <h4>Disclaimer</h4>
                    <p>This chatbot works only for educational purposes. For full medical assessment you should contact professionals.</p>
                </div>
            )}
        </div>
    );
}

export default Chatbot;