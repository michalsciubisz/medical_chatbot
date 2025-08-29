// export default Chatbot;
import React, { useState, useEffect, useRef, useCallback } from "react";
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
  const [isTyping, setIsTyping] = useState(false);
  const [validationByQ, setValidationByQ] = useState({});
  const [currentQ, setCurrentQ] = useState(null);
  const [askedCount, setAskedCount] = useState(0);
  const [useSlider, setUseSlider] = useState(false); // default OFF for a more "chatty" feel
  const [textRisk, setTextRisk] = useState(null); // 'low' | 'moderate' | 'high' | null


  const messagesEndRef = useRef(null);

  // ---- UI config ----
  const MAX_FILE_MB = 10;
  const ACCEPTED_MIME = ["image/jpeg", "image/png"];

  // auto-scroll
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // init
  useEffect(() => {
    const init = async () => {
      try {
        const res = await axios.post("http://localhost:5000/api/start");
        setSessionId(res.data.session_id);
        setMessages([
          {
            text:
              "Hello! Please choose type of assessment:\nType 'text' for questionnaire or 'image' for MRI analysis",
            sender: "bot",
          },
        ]);
        setAskedCount(0);
      } catch (e) {
        setMessages([{ text: "Failed to start session. Try again.", sender: "bot" }]);
      }
    };
    init();
  }, []);

  // keyboard shortcuts 0–9 for scale when input isn’t focused
  useEffect(() => {
    const onKey = (e) => {
      if (!currentQ || currentQ.type !== "scale") return;
      const active = document.activeElement;
      const isTextInput = active && active.tagName === "INPUT";
      if (isTextInput) return;

      if (e.key >= "0" && e.key <= "9") {
        const v = Number(e.key);
        const min = currentQ.meta?.min ?? 0;
        const max = currentQ.meta?.max ?? 4;
        if (v >= min && v <= max) {
          const labels = currentQ.meta?.labels || [];
          const label = labels[v - min] ?? String(v);
          handleScaleResponse(v, label);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [currentQ]);

  // remember slider pref
  useEffect(() => {
    const saved = localStorage.getItem("chatOA_useSlider");
    if (saved != null) setUseSlider(saved === "true");
  }, []);
  useEffect(() => {
    localStorage.setItem("chatOA_useSlider", String(useSlider));
  }, [useSlider]);

  // ---- NLU-lite ----

  // number words (0..10)
  const NUMBER_WORDS = {
    zero: 0, none: 0, "no": 0,
    one: 1, "a little": 1, slight: 1,
    two: 2, couple: 2, few: 2, mild: 1, // note: "mild" handled below in intensity
    three: 3, several: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8,
    nine: 9,
    ten: 10,
  };

  // agreement / polarity
  const YES_WORDS = [
    "yes", "yep", "yeah", "sure", "of course", "certainly", "affirmative", "yup", "ok", "okay", "definitely", "absolutely"
  ];
  const NO_WORDS = [
    "no", "nope", "not really", "rather not", "negative", "nah", "never"
  ];

  // intensity lexicon (mapped later to 0..3 or 0..4 depending on range)
  const INTENSITY = {
    none: ["none", "no pain", "no symptoms", "absent"],
    mild: ["mild", "slight", "a little", "tiny", "minor", "light"],
    moderate: ["moderate", "medium", "average", "tolerable"],
    severe: ["severe", "strong", "intense", "major", "significant", "bad", "quite bad"],
    extreme: ["very severe", "very strong", "very intense", "extreme", "unbearable", "worst", "maximal", "maximum"],
  };

  // frequency lexicon → also usable for Likert-like scales
  const FREQUENCY = {
    never: ["never"],
    rarely: ["rarely", "seldom", "occasionally"],
    sometimes: ["sometimes", "at times", "now and then"],
    often: ["often", "frequently", "regularly"],
    always: ["always", "constantly", "all the time"]
  };

  // degree adverbs (boosters)
  const DEGREE = {
    slightly: ["slightly", "a bit", "a little", "somewhat"],
    quite: ["quite", "fairly", "pretty"],
    very: ["very", "really"],
    extremely: ["extremely", "hugely", "tremendously", "incredibly"]
  };

  // ...

function normalizeRisk(val) {
  if (!val) return null;
  const t = String(val).toLowerCase();
  if (t.includes("high")) return "high";
  if (t.includes("moderate") || t.includes("medium")) return "moderate";
  if (t.includes("low")) return "low";
  return null;
}

function riskFromHeuristic(p) {
  // p ∈ [0,1]
  if (p >= 0.75) return "high";
  if (p >= 0.45) return "moderate";
  return "low";
}

function riskLabel(r) {
  if (r === "high") return "High OA likelihood";
  if (r === "moderate") return "Moderate OA likelihood";
  if (r === "low") return "Low OA likelihood";
  return "—";
}

function riskClass(r) {
  return r ? `risk-badge risk-${r}` : "risk-badge";
}


  function normalize(str) {
    return str.toLowerCase().replace(/\s+/g, " ").trim();
  }

  function containsAny(text, arr) {
    const t = normalize(text);
    return arr.some(w => t.includes(w));
  }

  function firstMatchIndex(text, groups) {
    // returns {key, index} for the first matched group in groups object
    const t = normalize(text);
    for (const [key, list] of Object.entries(groups)) {
      if (list.some(w => t.includes(w))) return key;
    }
    return null;
  }

  function extractNumberLike(text) {
    const t = normalize(text);

    // digits or fraction: "3", "3/10", "7 out of 10"
    const numMatch = t.match(/(\d+)\s*(?:\/|\bout of\b|\bna\b)?\s*(\d+)?/);
    if (numMatch) {
      const n = parseInt(numMatch[1], 10);
      const base = numMatch[2] ? parseInt(numMatch[2], 10) : null;
      if (Number.isFinite(n)) return { n, base };
    }

    // number words "three", "ten", "zero"
    for (const [w, v] of Object.entries(NUMBER_WORDS)) {
      if (t.includes(w)) return { n: v, base: null };
    }

    return null;
    }

  // map natural text → integer in [min,max]
  function mapTextToScale(text, min = 0, max = 4) {
    const t = normalize(text);

    // 1) direct numbers
    const numLike = extractNumberLike(t);
    if (numLike) {
      const { n, base } = numLike;
      if (Number.isFinite(n)) {
        if (base && base > 0) {
          const scaled = Math.round((n / base) * (max - min) + min);
          return clamp(scaled, min, max);
        }
        return clamp(n, min, max);
      }
    }

    // 2) frequency (never..always)
    const freq = firstMatchIndex(t, FREQUENCY);
    if (freq) {
      const map4 = { never: 0, rarely: 1, sometimes: 2, often: 3, always: 4 };
      const map3 = { never: 0, rarely: 1, sometimes: 2, often: 2, always: 3 }; // compress to 0..3
      return (max - min === 4)
        ? clamp(map4[freq] + min, min, max)
        : clamp(map3[freq] + min, min, max);
    }

    // 3) intensity core words
    const intensityKey =
      firstMatchIndex(t, {
        none: INTENSITY.none,
        extreme: INTENSITY.extreme,
        severe: INTENSITY.severe,
        moderate: INTENSITY.moderate,
        mild: INTENSITY.mild,
      });

    // degree modifiers
    const hasSlightly = containsAny(t, DEGREE.slightly);
    const hasQuite = containsAny(t, DEGREE.quite);
    const hasVery = containsAny(t, DEGREE.very);
    const hasExtremely = containsAny(t, DEGREE.extremely);

    if (intensityKey) {
      // base mapping for 0..4
      const base4 = { none: 0, mild: 1, moderate: 2, severe: 3, extreme: 4 };
      // base mapping for 0..3
      const base3 = { none: 0, mild: 1, moderate: 2, severe: 3, extreme: 3 };

      let val = (max - min === 4) ? base4[intensityKey] : base3[intensityKey];

      // apply degree shifts (small nudges within range)
      let shift = 0;
      if (hasSlightly) shift -= 0.3;
      if (hasQuite) shift += 0.3;
      if (hasVery) shift += 0.6;
      if (hasExtremely) shift += 1.0;

      const final = Math.round(val + shift);
      return clamp(final + min, min, max);
    }

    // 4) yes/no as binary scale (if question is likely binary)
    if (YES_WORDS.some(w => t.includes(w))) return clamp(min + (max > min ? 1 : 0), min, max);
    if (NO_WORDS.some(w => t.includes(w))) return clamp(min, min, max);

    return null;
  }

  function clamp(n, min, max) {
    return Math.min(max, Math.max(min, n));
  }

  function detectIntent(text) {
    const t = normalize(text);
    // keep only intents that don't duplicate top UI:
    // - We REMOVE finish/help here (buttons exist at the top)
    if (t.startsWith("/restart") || t.includes("restart")) return "RESTART";
    if (t.includes("image mode") || t.includes("mri mode") || t.startsWith("/image")) return "IMAGE_MODE";
    if (t.includes("upload image") || t.includes("send image")) return "UPLOAD_IMAGE";
    if (t.includes("skip") || t.startsWith("/skip")) return "SKIP";
    if (t.includes("undo") || t.includes("edit last")) return "UNDO";
    if (t.includes("summary") || t.startsWith("/summary")) return "SUMMARY";
    return null;
  }

  // input is always enabled
  const isInputDisabled = false;

  // --- Question components ---
  const ScaleQuestion = ({ m }) => {
    const labels = m.meta?.labels || ["0", "1", "2", "3", "4"];
    const min = m.meta?.min ?? 0;
    const max = m.meta?.max ?? 4;
    const options = Array.from({ length: max - min + 1 }, (_, i) => i + min);
    const vMsg = validationByQ[m.question_id];

    const [sliderVal, setSliderVal] = useState(min);

    const submitSlider = () => {
      const num = Number(sliderVal);
      const idx = num - min;
      const label = labels[idx] ?? String(num);
      handleScaleResponse(num, label);
    };

    return (
      <div className="scale-question" role="radiogroup" aria-label={m.question}>
        <p className="question-text">
          {m.question}
          <span
            className="why-tip"
            title={m.meta?.why || "This question helps estimate severity and risk."}
            style={{ marginLeft: 6, cursor: "help" }}
          >
            ⓘ
          </span>
        </p>

        <div className="rating-scale">
          {options.map((v, idx) => {
            const label = labels[idx] ?? v;
            return (
              <button
                key={idx}
                onClick={() => handleScaleResponse(v, label)}
                className="scale-option"
                aria-pressed="false"
                aria-label={`Value ${v} - ${label}`}
              >
                <span className="number">{v}</span>
                <span className="label">{label}</span>
              </button>
            );
          })}
        </div>

        {useSlider && (
          <div className="slider-row">
            <input
              type="range"
              min={min}
              max={max}
              step="1"
              value={sliderVal}
              onChange={(e) => setSliderVal(Number(e.target.value))}
              aria-label="Scale slider"
            />
            <div className="slider-value">
              <span className="number">{sliderVal}</span>
              <span className="label">{labels[sliderVal - min] ?? sliderVal}</span>
            </div>
            <button className="go-btn" onClick={submitSlider}>
              Select
            </button>
          </div>
        )}

        {vMsg && <div className="validation-hint">{vMsg}</div>}
        <p className="kb-hint">Tip: you can also write “mild”, “severe”, “3/10”, or press keys {min}–{max}.</p>
      </div>
    );
  };

  const ChoiceQuestion = ({ m }) => {
    const vMsg = validationByQ[m.question_id];
    return (
      <div className="choice-question" role="group" aria-label={m.question}>
        <p className="question-text">
          {m.question}
          <span
            className="why-tip"
            title={m.meta?.why || "Selecting an option helps tailor the next questions."}
            style={{ marginLeft: 6, cursor: "help" }}
          >
            ⓘ
          </span>
        </p>
        <div className="choices">
          {(m.meta?.choices || []).map((c) => (
            <button key={c} onClick={() => submitAnswer(c, c)}>
              {c}
            </button>
          ))}
        </div>
        {vMsg && <div className="validation-hint">{vMsg}</div>}
      </div>
    );
  };

  const FreeTextQuestion = ({ m }) => {
    const vMsg = validationByQ[m.question_id];
    return (
      <div className="freetext-question">
        <p className="question-text">
          {m.question}
          <span
            className="why-tip"
            title={m.meta?.why || "Tell me in your own words; I can work with free text."}
            style={{ marginLeft: 6, cursor: "help" }}
          >
            ⓘ
          </span>
        </p>
        {vMsg && <div className="validation-hint">{vMsg}</div>}
      </div>
    );
  };

  const renderQuestion = (m) => {
    if (m.sender === "bot" && m.question) {
      if (m.type === "scale") return <ScaleQuestion m={m} />;
      if (m.type === "choice") return <ChoiceQuestion m={m} />;
      return <FreeTextQuestion m={m} />;
    }
    return <p>{m.text}</p>;
  };

  // --- send answer ---
  const submitAnswer = async (valueOverride, userEchoText) => {
    if (!sessionId) return;

    const payload = {
      session_id: sessionId,
      answer: valueOverride !== undefined ? String(valueOverride) : input.trim(),
    };
    if (!payload.answer) return;

    // optimistic echo
    if (valueOverride === undefined) {
      setMessages((prev) => [...prev, { text: payload.answer, sender: "user" }]);
      setInput("");
    } else {
      const echo = userEchoText ?? String(valueOverride);
      setMessages((prev) => [...prev, { text: echo, sender: "user" }]);
    }

    try {
      setIsTyping(true);
      const res = await axios.post("http://localhost:5000/api/respond", payload);
      const data = res.data;

      if (data.text_risk) {
        setTextRisk(normalizeRisk(data.text_risk) || null);
      }
      
      if (!data.mode && data.result && typeof data.result === "string") {
        const inferred = normalizeRisk(data.result);
        if (inferred) setTextRisk(inferred);
      }

      if (data.mode === "image") setImageModeActive(true);

      if (currentQ?.question_id) {
        setValidationByQ((prev) => {
          const copy = { ...prev };
          delete copy[currentQ.question_id];
          return copy;
        });
      }

      if (data.result) {
        setMessages((prev) => [...prev, { text: data.result, sender: "bot" }]);
        setIsExamComplete(true);
        setIsTyping(false);
        return;
      }

      if (data.question) {
        const botMsg = {
          sender: "bot",
          question: data.question,
          question_id: data.question_id,
          type: data.type,
          meta: data.meta,
        };
        setMessages((prev) => [...prev, botMsg]);
        setCurrentQ(botMsg);
        setAskedCount((c) => c + 1);
        setIsTyping(false);
        return;
      }

      if (data.mode === "image") {
        setMessages((prev) => [
          ...prev,
          { text: "Please upload your MRI image using the upload section below.", sender: "bot" },
        ]);
      }
      setIsTyping(false);
    } catch (err) {
      setIsTyping(false);
      const data = err?.response?.data;
      if (data?.error === "validation_error") {
        if (data.question_id) {
          setValidationByQ((prev) => ({
            ...prev,
            [data.question_id]: data.message || "Invalid answer.",
          }));
        }
        const botMsg = {
          sender: "bot",
          question: data.question,
          question_id: data.question_id,
          type: data.type,
          meta: data.meta,
        };
        setMessages((prev) => [...prev, botMsg]);
        setCurrentQ(botMsg);
      } else {
        setMessages((prev) => [
          ...prev,
          { text: "Sorry, I encountered an error. Please try again.", sender: "bot" },
        ]);
      }
    }
  };

  const handleScaleResponse = (value, label) => {
    const echo = label ? `${value} – ${label}` : String(value);
    submitAnswer(value, echo);
  };

  const restartConversation = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/start");
      setSessionId(res.data.session_id);
      setMessages([
        {
          text:
            "Hello! Please choose type of assessment:\nType 'text' for questionnaire or 'image' for MRI analysis",
          sender: "bot",
        },
      ]);
      setResult(null);
      setImage(null);
      setIsExamComplete(false);
      setEarlyFinishWarning(false);
      setImageModeActive(false);
      setValidationByQ({});
      setCurrentQ(null);
      setAskedCount(0);
    } catch (error) {
      setMessages((prev) => [...prev, { text: "Error restarting conversation.", sender: "bot" }]);
    }
  };

  const handleEarlyFinish = async () => {
    if (!sessionId) return;
    try {
      setIsTyping(true);
      const res = await axios.post("http://localhost:5000/api/early_finish", {
        session_id: sessionId,
      });
      const data = res.data;

      if (data.text_risk) {
        setTextRisk(normalizeRisk(data.text_risk) || null);
      } else if (data.result && typeof data.result === "string") {
        const inferred = normalizeRisk(data.result);
        if (inferred) setTextRisk(inferred);
      }

      if (data.result) {
        setMessages((prev) => [
          ...prev,
          { text: data.result, sender: "bot" },
          { text: data.message, sender: "bot" },
        ]);
        setIsExamComplete(true);
      } else if (data.low_confidence) {
        setMessages((prev) => [...prev, { text: data.message, sender: "bot" }]);
      }
      setIsTyping(false);
      setEarlyFinishWarning(false);
    } catch (error) {
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        { text: "Error completing assessment early. Please try again.", sender: "bot" },
      ]);
    }
  };

  // Image: drag & drop + validation
  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) validateAndSetImage(f);
    },
    []
  );
  const onDragOver = (e) => e.preventDefault();

  const validateAndSetImage = (file) => {
    if (!ACCEPTED_MIME.includes(file.type)) {
      alert("Please upload a PNG or JPEG image.");
      return;
    }
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_MB) {
      alert(`Max file size is ${MAX_FILE_MB} MB.`);
      return;
    }
    setImage(file);
  };

  const handleImageUpload = (e) => {
    const f = e.target.files?.[0];
    if (f) validateAndSetImage(f);
  };

  const submitImage = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("image", image);
    try {
      setIsTyping(true);
      const response = await axios.post(
        "http://localhost:5000/api/classify_image",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(response.data.prediction);
      setMessages((prev) => [
        ...prev,
        {
          text: `Based on MRI imaging you are likely to have: ${response.data.prediction}.`,
          sender: "bot",
        },
      ]);
      setIsTyping(false);
    } catch (error) {
      setIsTyping(false);
      setResult("Error: Could not classify image.");
      setMessages((prev) => [...prev, { text: "Image classification failed. Try again.", sender: "bot" }]);
    }
  };

  const heurConfidence = () => Math.min(0.95, 0.35 + 0.05 * askedCount);

  const copySummary = () => {
    const text = [
      "Summary of results:",
      `- Based on text interview: ${isExamComplete ? "assessment complete" : "incomplete"}`,
      `- Based on MRI imaging: ${result || "No image analyzed yet"}`,
      `- Heuristic confidence: ${(heurConfidence() * 100).toFixed(0)}%`,
      "",
      "Next steps:",
      "1) Book a consultation with a GP/orthopedist.",
      "2) Consider physiotherapy (ROM & strengthening exercises).",
      "3) Low-impact activity (swim, bike), weight management.",
      "4) Discuss analgesics/NSAIDs with a clinician.",
      "5) Re-check symptoms in 4–6 weeks.",
    ].join("\n");
    navigator.clipboard.writeText(text);
    alert("Summary copied to clipboard.");
  };

  // send with interpretation & intents
  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    // intents (keep only ones that make sense — finish/help are in the top bar)
    const intent = detectIntent(trimmed);
    if (intent === "RESTART") {
      await restartConversation();
      setInput("");
      return;
    }
    if (intent === "IMAGE_MODE") {
      setImageModeActive(true);
      setActiveTab("chat");
      setMessages((prev) => [...prev, { sender: "bot", text: "Image mode enabled. Upload your MRI below." }]);
      setInput("");
      return;
    }
    if (intent === "UPLOAD_IMAGE") {
      setImageModeActive(true);
      setActiveTab("chat");
      setInput("");
      return;
    }
    if (intent === "SKIP" && currentQ) {
      await submitAnswer("skip", "Skip");
      setInput("");
      return;
    }
    if (intent === "UNDO") {
      const lastUser = [...messages].reverse().find((m) => m.sender === "user");
      if (lastUser?.text) setInput(lastUser.text);
      return;
    }
    if (intent === "SUMMARY") {
      setActiveTab("results");
      setInput("");
      return;
    }

    // NLU-lite for current question
    if (currentQ) {
      if (currentQ.type === "scale") {
        const min = currentQ.meta?.min ?? 0;
        const max = currentQ.meta?.max ?? 4;
        const parsed = mapTextToScale(trimmed, min, max);
        if (parsed !== null) {
          const label = currentQ.meta?.labels?.[parsed - min] ?? String(parsed);
          await submitAnswer(parsed, `${parsed} – ${label}`);
          setInput("");
          return;
        }
      }
      if (currentQ.type === "choice") {
        const choices = (currentQ.meta?.choices || []);
        // direct text includes choice
        const found = choices.find((c) => trimmed.toLowerCase().includes(c.toLowerCase()));
        if (found) {
          await submitAnswer(found, found);
          setInput("");
          return;
        }
        // yes/no to binary choices if applicable
        const t = trimmed.toLowerCase();
        const lowerChoices = choices.map((c) => c.toLowerCase());
        if (YES_WORDS.some((w) => t.includes(w)) && lowerChoices.includes("yes")) {
          await submitAnswer("yes", "yes");
          setInput("");
          return;
        }
        if (NO_WORDS.some((w) => t.includes(w)) && lowerChoices.includes("no")) {
          await submitAnswer("no", "no");
          setInput("");
          return;
        }
      }
    }

    // fallback: send raw to backend (server keeps validation)
    await submitAnswer();
  };

  return (
    <div className="chatbot-container">
      <header className="chatbot-header">
        <div className="header-content">
          <img src="logo.png" alt="Medical Logo" className="header-logo" />
        <div>
            <h1>chatOA</h1>
            <p>chatbot osteoarthritis interpretation</p>
          </div>
        </div>
        <p className="subtitle">Check if you have osteoarthritis!</p>
      </header>

      <div className="chatbot-tabs">
        <button className={activeTab === "chat" ? "active" : ""} onClick={() => setActiveTab("chat")}>Chat</button>
        <button className={activeTab === "results" ? "active" : ""} onClick={() => setActiveTab("results")}>Results</button>
        <button className={activeTab === "info" ? "active" : ""} onClick={() => setActiveTab("info")}>About OA</button>
      </div>

      {activeTab === "chat" && (
        <div className="chat-section">
          <div className="chat-actions">
            <button onClick={restartConversation} aria-label="Restart assessment">Restart</button>
            <button onClick={() => setEarlyFinishWarning(true)} aria-label="Finish early">Finish now</button>
            <button onClick={() => setActiveTab("info")} aria-label="Help">Help</button>

            <label className="toggle">
              <input
                type="checkbox"
                checked={useSlider}
                onChange={(e) => setUseSlider(e.target.checked)}
              />
              <span>Show slider (assistive)</span>
            </label>
          </div>

          {earlyFinishWarning && (
            <div role="dialog" aria-modal="true" className="warning-message">
              <p>You may have too few answers for a reliable assessment. Finish anyway?</p>
              <div>
                <button onClick={() => setEarlyFinishWarning(false)}>Continue</button>
                <button onClick={handleEarlyFinish}>Finish anyway</button>
              </div>
            </div>
          )}

          <div className="messages-container">
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.sender}`}>{renderQuestion(m)}</div>
            ))}
            {isTyping && (
              <div className="message bot typing">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Smart suggestions (only the ones that add value) */}
          <div className="smart-suggestions" style={{ display: "flex", gap: 8, margin: "6px 0" }}>
            <button onClick={() => setInput("skip")} className="smart-chip">Skip question</button>
            {imageModeActive ? (
              <button onClick={() => setInput("upload image")} className="smart-chip">Upload image</button>
            ) : (
              <button onClick={() => setInput("image mode")} className="smart-chip">Image mode</button>
            )}
            <button onClick={() => setInput("summary")} className="smart-chip">Summary</button>
          </div>

          <div className="input-area">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={"Type freely… e.g., “pain is quite severe when climbing stairs”"}
              aria-disabled={false}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
            />
            <button onClick={handleSend}>Send</button>
          </div>

          {imageModeActive && (
            <div className="image-upload-section">
              <h3>Provide your MRI scan:</h3>
              <div className="upload-box" onDrop={onDrop} onDragOver={onDragOver}>
                {image ? (
                  <div className="upload-preview">
                    <div className="preview-left">
                      <span className="file-name">{image.name}</span>
                      <span className="file-size">{(image.size / (1024 * 1024)).toFixed(2)} MB</span>
                    </div>
                    <div className="preview-actions">
                      <button onClick={() => setImage(null)}>Remove</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p>Drop image here or choose a file</p>
                    <input type="file" accept="image/png,image/jpeg" onChange={handleImageUpload} id="image-upload" />
                    <label htmlFor="image-upload">Choose File</label>
                    <p className="hint">PNG or JPEG, up to {MAX_FILE_MB} MB</p>
                  </>
                )}
              </div>
              {image && (
                <button className="classify-button" onClick={submitImage}>
                  {isTyping ? "Analyzing…" : "Classify image"}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === "results" && (
        <div className="results-section">
          <h2>Summary of results</h2>
          <div className="results-grid">
            {/* <section className="card">
              <h3>Overall result</h3>
              <div className="risk-badge risk-moderate">Moderate OA likelihood</div>
              <p className="confidence">Confidence (heuristic): {(heurConfidence() * 100).toFixed(0)}%</p>
            </section> */}
            <section className="card">
              <h3>Overall result</h3>
              {(() => {
                const fallback = riskFromHeuristic(heurConfidence());
                const r = textRisk || fallback;
                return (
                  <div className={riskClass(r)}>
                    {riskLabel(r)}
                  </div>
                );
              })()}
              <p className="confidence">Confidence (heuristic): {(heurConfidence() * 100).toFixed(0)}%</p>
            </section>

            <section className="card">
              <h3>Interview (text)</h3>
              <p>{isExamComplete ? "Assessment complete based on questionnaire." : "Not enough data for a reliable assessment."}</p>
              <ul className="bullets">
                <li>Scale responses summarized in the conversation.</li>
                <li>Higher pain/stiffness scores increase OA likelihood.</li>
              </ul>
            </section>

            <section className="card">
              <h3>MRI analysis</h3>
              <p>{result ? `Predicted: ${result}` : "No image analyzed yet."}</p>
            </section>

            <section className="card">
              <h3>What you can do next</h3>
              <ol className="next-steps">
                <li>Consult your GP or an orthopedist with these results.</li>
                <li>Consider physiotherapy (ROM & strengthening exercises).</li>
                <li>Low-impact activity (swimming, cycling), weight management.</li>
                <li>Discuss analgesic/anti-inflammatory options with a clinician.</li>
                <li>Re-assess symptoms in 4–6 weeks.</li>
              </ol>
              <div className="resources">
                <span>Helpful resources:</span>
                <ul>
                  <li>WHO – Musculoskeletal health (patient overview)</li>
                  <li>NHS – Osteoarthritis: symptoms, diagnosis, treatment</li>
                  <li>Exercise library for OA (physio-friendly routines)</li>
                </ul>
              </div>
              <div className="results-actions">
                <button onClick={copySummary}>Copy summary</button>
              </div>
              <p className="disclaimer">
                This tool is for educational purposes only and is not a medical diagnosis. Always consult a healthcare professional.
              </p>
            </section>
          </div>
        </div>
      )}

      {activeTab === "info" && (
        <div className="info-section">
          <h2>About Osteoarthritis</h2>
          <p>Osteoarthritis (OA) is the most common form of arthritis, affecting millions of people worldwide.</p>
          <h3>How does this chatbot work?</h3>
          <p>
            The chatbot analyzes your symptoms through guided questions and can evaluate MRI images.
            Based on your answers and image classification, it provides an educational summary and next-step recommendations.
          </p>
          <h4>Disclaimer</h4>
          <p>This chatbot is for educational purposes only. For a full medical assessment, please contact healthcare professionals.</p>
        </div>
      )}
    </div>
  );
}

export default Chatbot;
