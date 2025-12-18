# 🕵️ ChatStalkerAI

A fun, minimalistic RAG + Analytics app for analyzing your WhatsApp group chat exports.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![Gemini](https://img.shields.io/badge/Gemini-AI-green)

## ✨ Features

### 💬 AI Chat Assistant (RAG)
Ask natural language questions about your group chat:
- "Summarize what everyone talked about"
- "Who talks about food the most?"
- "What was the plan for the Paris trip?"

### 📊 Word Counter
Analyze word usage in your chat:
- Search for any word or phrase
- Filter by participant or view everyone
- Beautiful bar chart visualization

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Keep it handy for the app

### 3. Export Your WhatsApp Chat

1. Open WhatsApp group chat
2. Tap ⋮ (menu) → More → Export chat
3. Choose "Without media"
4. Save the `.txt` file

### 4. Run the App

```bash
streamlit run app.py
```

### 5. Use the App

1. Upload your WhatsApp `.txt` export in the sidebar
2. Enter your Gemini API key
3. Switch between tabs:
   - **Chat Assistant**: Ask AI questions
   - **Word Counter**: Search for words

## 📁 Project Structure

```
chatStalkerAI/
├── app.py              # Main Streamlit app
├── utils/
│   ├── parser.py       # WhatsApp chat parser
│   ├── embeddings.py   # RAG with Gemini
│   └── analytics.py    # Word count analytics
├── requirements.txt
└── README.md
```

## 🎨 Features in Detail

### Chat Parser
- Handles multiple WhatsApp date formats
- Filters out system messages
- Extracts sender, timestamp, and message

### Word Counter
- Case-insensitive search
- Participant filtering
- Visual bar chart when viewing "All"

### RAG Chat Assistant
- Uses Gemini Pro for responses
- Simple keyword-based retrieval
- Fun, engaging response style

## 🔒 Privacy

Your chat data is:
- Processed locally on your machine
- Only sent to Gemini API when you ask questions
- Never stored permanently

## 📝 License

MIT License - Use it however you want!

---

Made with 💚 and curiosity

