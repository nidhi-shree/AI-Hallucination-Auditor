# FactVibe: AI Hallucination Auditor

> **Detect hallucinations and verify factual claims using AI and real-world evidence.**

FactVibe is a production-quality Streamlit application that helps users identify hallucinations and factual inaccuracies in AI-generated content. It combines the reasoning power of **Groq (Qwen3-32B)** with live evidence from **DuckDuckGo Search** to deliver an objective, evidence-backed audit report.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Claim Extraction** | Groq (Qwen3-32B) automatically extracts 3–7 verifiable factual claims |
| 🌐 **Evidence Retrieval** | DuckDuckGo fetches real-world search results for each claim |
| ✅ **Claim Verification** | Groq cross-references claims against evidence |
| 📊 **Hallucination Score** | Visual risk score: Low / Moderate / High |
| 📄 **PDF Export** | Download a full audit report as a PDF |
| 🎨 **Premium UI** | Dark glassmorphism dashboard with animated metric cards |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Groq API** (`groq` SDK) — Claim extraction & verification (model: `qwen/qwen3-32b`, fallback: `llama-3.3-70b-versatile`)
- **DuckDuckGo Search** (`duckduckgo-search`) — External evidence retrieval
- **ReportLab** — PDF report generation
- **python-dotenv** — Environment variable management

---

## 🚀 Quick Start

### 1. Clone / Download the Project

```bash
# Navigate to your project directory
cd genAI_assignment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the example env file
copy .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get your free API key at [Groq Console](https://console.groq.com/)

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Application Workflow

```
User Input Text
      │
      ▼
┌─────────────────────────┐
│  Step 1: Claim Extraction│  ← Groq (Qwen3-32B) extracts 3-7 facts
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Step 2: Evidence Search │  ← DuckDuckGo retrieves top 3 results
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Step 3: Verification   │  ← Groq classifies each claim
└─────────────────────────┘
      │
      ▼
  Audit Report + PDF Export
```

---

## 📊 Verdict Classification

| Verdict | Meaning |
|---|---|
| ✅ **Verified** | Evidence clearly supports the claim |
| 🟡 **Partially Supported** | Evidence is inconclusive or partial |
| ❓ **Unverified** | No relevant evidence found |
| ❌ **Contradicted** | Evidence clearly refutes the claim |

---

## 📈 Hallucination Score

```
Hallucination Score = (Unverified + Contradicted) / Total Claims × 100
```

| Score | Risk Level | Color |
|---|---|---|
| 0–20% | 🟢 Low Risk | Green |
| 21–50% | 🟡 Moderate Risk | Amber |
| 51–100% | 🔴 High Risk | Red |

---

## 📁 Project Structure

```
genAI_assignment/
├── app.py              # Main application (single-file architecture)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your local env file (not committed)
├── styles.css          # Custom CSS for the UI
└── README.md           # This file
```

---

## ⚡ Function Reference

| Function | Description |
|---|---|
| `call_llm(prompt)` | Reusable helper — sends prompts to Groq with retry handling |
| `get_groq_client()` | Initialises the singleton Groq client (probes primary model, falls back if needed) |
| `extract_claims(text)` | Uses Groq to extract verifiable facts from text |
| `search_evidence(claim)` | Searches DuckDuckGo for evidence per claim |
| `verify_claim(claim, evidence)` | Uses Groq to classify claim vs evidence |
| `calculate_hallucination_score()` | Computes the overall risk score |
| `generate_pdf_report()` | Generates a downloadable PDF audit report |
| `main()` | Streamlit app entry point |

---

## 🔧 Configuration

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes |

---

## 🔄 Model Selection

| Priority | Model | Notes |
|---|---|---|
| Primary | `qwen/qwen3-32b` | Default — fast, high-quality reasoning |
| Fallback | `llama-3.3-70b-versatile` | Used if primary model is unavailable |

The active model is resolved automatically at startup. No manual configuration required.

---

## 🔁 Retry Handling

`call_llm()` retries automatically on:

| Error Type | Behaviour |
|---|---|
| `RateLimitError` | Exponential back-off (base 10 s, ×1.5 per attempt, up to 5 retries) |
| `APIStatusError` (5xx) | Same back-off policy |
| `APITimeoutError` | Same back-off policy |
| Client errors (4xx) | Surfaced immediately — no retry |

---

## ⚠️ Disclaimer

FactVibe is an AI-assisted tool and may not be 100% accurate. The verdicts are based on available web evidence and AI reasoning. Always verify critical claims with authoritative primary sources.

---

## 📜 License

MIT License — free for personal and educational use.

---

*Built with ❤️ using Groq (Qwen3-32B) + Streamlit*
