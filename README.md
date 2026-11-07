# 🌍 Lingilo - Your Personal AI Language Assistant

**Lingilo** is an advanced, AI-powered web application built with Streamlit, designed to comprehensively support foreign language learning. It combines traditional memorization techniques (Spaced Repetition) with the power of Large Language Models (LLMs), semantic search, and speech transcription. 

Lingilo is not just another translator – it is an intelligent environment that helps you understand context, analyze grammar, practice pronunciation, and systematically acquire new vocabulary.

---

## 🚀 Core Features & Learning Modules

The application offers 6 main modes to support users at every stage of their learning journey:

1. 📚 **Smart Flashcards & Spaced Repetition System (SM-2)**
   * The app features a built-in Spaced Repetition algorithm (SM-2) that optimizes learning time by showing words exactly when you are about to forget them.
   * **AI Flashcard Creator:** Just type a single word, and the AI will automatically generate its base form, grammatical features (case, tense, gender), definitions, inflections, and example sentences with translations.
   * **Authentic Context:** Integration with the open **Tatoeba** database allows you to fetch real-life example sentences with a single click.

2. ✍️ **Text Correction with Explanations**
   * Paste a text written in a foreign language, and the AI will not only correct grammatical and stylistic errors but also explain point-by-point *why* the changes were made and what caused the mistakes.

3. 🗣️ **Pronunciation Training & Audio Generation (TTS)**
   * Convert any text or translated sentence into natural-sounding speech (Text-To-Speech) using the `gTTS` library.
   * Ability to record your own pronunciation and attach it directly to your flashcards.

4. 🎙️ **Voice Notes with Transcription**
   * Record a voice note in a foreign language, and the system will automatically transcribe it into text. Great for practicing spontaneous speaking.

5. 🧠 **Knowledge Base with Semantic Search**
   * All translations, expert queries, and voice notes are saved in the Database.
   * Searches are performed "by meaning" (vector-based), meaning if you search for "pets," the system will find a note containing "dog" or "cat," even if the exact keyword isn't present in the text.

6. 💬 **Chat with a Language Expert**
   * A dedicated module to ask any linguistic question (e.g., *"What is the difference between 'make' and 'do' in English?"*) and receive a comprehensive, educational explanation from the AI.

---

## 🤖 AI Models Used & Their Roles

Lingilo is a truly *AI-Native* system. Under the hood, it uses various models and technologies for specific tasks:

### 1. Large Language Models (LLMs) - Groq & OpenAI
The app allows users to choose between lightning-fast, free open-source models via the **Groq** cloud (e.g., the Llama 3 family) and advanced paid models from **OpenAI** (e.g., `gpt-4o`). If a paid API key is missing, the system has a built-in, safe fallback to the free Groq model.
* **What is it used for?** 
  * Providing precise translations while preserving context.
  * Correcting texts and generating educational explanations of errors.
  * Powering the "Language Expert" chat mode.

### 2. `Instructor` + Pydantic (Structured Outputs)
LLMs naturally generate unstructured text. To safely add words to the database, `Instructor` forces the AI to output a strict, validated JSON based on Pydantic models.
* **What is it used for?** 
  * Parsing LLM output into a predictable structure when creating flashcards: extracting sentence arrays and dictionaries of grammatical features (e.g., extracting tense, number, person, case).

### 3. Whisper Model (`whisper-large-v3`)
An advanced multilingual speech recognition model (run asynchronously via OpenAI / Groq API).
* **What is it used for?** 
  * Transcribing audio recordings and user voice notes into clean text.

### 4. Vector Models (`sentence-transformers`)
The system uses a locally loaded `paraphrase-multilingual-MiniLM-L12-v2` model.
* **What is it used for?** 
  * Generating *embeddings* (multidimensional vectors) from text. This powers the advanced **semantic search engine** for the app's notes.

### 5. AI Telemetry - Langfuse
Integration with the **Langfuse** platform (using the direct HTTP API `generation-create`).
* **What is it used for?**
  * Tracking every AI trace, counting used tokens (`promptTokens`, `completionTokens`), monitoring costs, and evaluating generation quality (latency, potential errors).

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Streamlit OAuth (Auth0 login)
* **Backend:** Python (Asynchronous Core: `asyncio`)
* **Database:** Asynchronous SQLAlchemy + PostgreSQL (compatible with Neon DB) or SQLite for local setups.
* **AI Tools:** LiteLLM (universal API router), Instructor, Sentence-Transformers.
