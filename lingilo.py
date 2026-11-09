import sys
import os
import io
import re
import json
import asyncio
import time
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field
from typing import List, Optional

import streamlit as st
import pandas as pd
import requests
import numpy as np
from dotenv import dotenv_values, load_dotenv

from sqlalchemy import Boolean, Column, Integer, Float, LargeBinary, String, Text, TIMESTAMP, func, Enum, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.future import select
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from gtts import gTTS
from langdetect import detect
import langcodes
import enum
import certifi
from streamlit_oauth import OAuth2Component  
from sentence_transformers import SentenceTransformer

# --- LANGFUSE, LITELLM I INSTRUCTOR ---
import openai
import litellm
from litellm import acompletion
import instructor
import uuid
import base64
import extra_streamlit_components as stx
from http.cookies import SimpleCookie

# Wyciszamy wewnętrzne logowanie LiteLLM żeby nie wywalało błędu Task was destroyed
litellm.success_callback = []
litellm.failure_callback = []

# --- NAPRAWA BŁĘDÓW SSL I ASYNCIO W SYSTEMIE WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ['SSL_CERT_FILE'] = certifi.where()

# ==========================================
# KONFIGURACJA I STAN SESJI
# ==========================================
env = dotenv_values(".env")
load_dotenv()

def get_secret(key, default_val=""):
    if key in env and env[key]: return env[key]
    if os.getenv(key): return os.getenv(key)
    try: return st.secrets[key]
    except Exception: return default_val

AUTH0_DOMAIN = get_secret("AUTH0_DOMAIN")  
CLIENT_ID = get_secret("CLIENT_ID")         
CLIENT_SECRET = get_secret("CLIENT_SECRET") 
REDIRECT_URI = get_secret("REDIRECT_URI", "http://localhost:8501/component/streamlit_oauth.authorize_button")

for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST",
          "GROQ_API_KEY", "OPENAI_API_KEY"]:
    val = get_secret(k)
    if val:
        os.environ[k] = val

oauth2 = OAuth2Component(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
    authorize_endpoint=f"https://{AUTH0_DOMAIN}/authorize",
    token_endpoint=f"https://{AUTH0_DOMAIN}/oauth/token",
    refresh_token_endpoint=f"https://{AUTH0_DOMAIN}/oauth/token",
    revoke_token_endpoint=f"https://{AUTH0_DOMAIN}/oauth/revoke"
)

# ==========================================
# PANCERNA PAMIĘĆ FORMULARZY
# ==========================================
def init_session_state():
    if "p_state" not in st.session_state:
        st.session_state.p_state = {
            'trans_text': "", 'corr_text': "", 'tts_text': "", 'inq_text': "",
            'note_audio_text': "", 'flashcard_input_word': "", 'fc_base_form': "",
            'fc_grammar_info': "", 'fc_forms': "", 'fc_custom_defs': "",
            'fc_custom_sents': "", 'fc_custom_notes': "",
            'fc_selected_defs': [], 'fc_selected_sents': [],
            'search_flashcards': "", 'search_notes': ""  
        }

    defaults = {
        'ui_language': "Polski", 'translation_target_language': "Angielski",
        'ai_provider': "Groq (Darmowe chmurowe)", 'active_task': "Tłumaczenie",
        'current_user_id': None, 'prefs_loaded': False, 'last_saved_drafts': None,
        
        'translated_text': "", 'corrected_text': "", 'inquiry_response': "",
        'note_audio_bytes': None, 'audio_generation_audio_file': None,
        
        'show_flashcard_answer': False, 'current_flashcard_draft': False,
        'flashcard_audio_bytes': None, 'flashcard_sub_tab': "study",
        
        'fc_def_options': [], 'fc_sent_options': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def p_text_area(label, key, **kwargs):
    def update(): st.session_state.p_state[key] = st.session_state[f"wid_{key}"]
    return st.text_area(label, key=f"wid_{key}", value=st.session_state.p_state.get(key, ""), on_change=update, **kwargs)

def p_text_input(label, key, **kwargs):
    def update(): st.session_state.p_state[key] = st.session_state[f"wid_{key}"]
    return st.text_input(label, key=f"wid_{key}", value=st.session_state.p_state.get(key, ""), on_change=update, **kwargs)

def p_multiselect(label, key, options, **kwargs):
    def update(): st.session_state.p_state[key] = st.session_state[f"wid_{key}"]
    safe_default = [x for x in st.session_state.p_state.get(key, []) if x in options]
    return st.multiselect(label, key=f"wid_{key}", options=options, default=safe_default, on_change=update, **kwargs)

def _(text): 
    lang = st.session_state.ui_language
    if not text.strip(): return text
    if lang == "Polski":
        polskie_enumy = {
            "TRANSLATION": "Tłumaczenie", "CORRECTION": "Korekta",
            "TEXT_TO_SPEECH": "Generowanie audio", "INQUIRY": "Zapytanie",
            "REVISION": "Powtórki", "TEXT": "Tekst", "AUDIO": "Audio",
            "GENERATED_AUDIO": "Wygenerowane TTS"
        }
        return polskie_enumy.get(text, text)
    return STATIC_TRANSLATIONS.get(lang, STATIC_TRANSLATIONS.get("Angielski", {})).get(text, text)

# ==========================================
# BAZA DANYCH (NEON / POSTGRESQL)
# ==========================================
def get_engine():
    db_url = get_secret("DATABASE_URL", "sqlite+aiosqlite:///lingilo_local.db")
    if "?" in db_url: db_url = db_url.split("?")[0]
    connect_args = {"ssl": True} if "postgresql" in db_url else {}
    return create_async_engine(db_url, echo=False, connect_args=connect_args, poolclass=NullPool)

Base = declarative_base()

class TaskType(enum.Enum):
    TRANSLATION = "translation"
    CORRECTION = "correction"
    TEXT_TO_SPEECH = "text_to_speech"
    INQUIRY = "inquiry"
    REVISION = "revision"

class NoteType(enum.Enum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"
    GENERATED_AUDIO = "GENERATED_AUDIO"

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    user_id = Column(String, primary_key=True)
    ui_language = Column(String, default="Polski")
    translation_target_language = Column(String, default="Angielski")
    ai_provider = Column(String, default="Groq (Darmowe chmurowe)")
    session_drafts = Column(Text, default="{}")

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    task_type = Column(Enum(TaskType, name="task_type"), nullable=False)
    input_text = Column(Text)
    output_text = Column(Text)

class WordRevision(Base):
    __tablename__ = "word_revisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    word = Column(String, nullable=False)
    base_form = Column(String, nullable=True) 
    grammar_info = Column(String, nullable=True) 
    definitions = Column(Text, nullable=True) 
    language = Column(String, nullable=False)
    inflected_forms = Column(Text, nullable=True)
    example_sentences = Column(Text, nullable=True)
    custom_notes = Column(Text, nullable=True) 
    audio_data = Column(LargeBinary, nullable=True) 
    creation_date = Column(TIMESTAMP, server_default=func.now())
    interval = Column(Integer, default=0)
    repetition = Column(Integer, default=0)
    easiness_factor = Column(Float, default=2.5)
    next_review_date = Column(TIMESTAMP, server_default=func.now())

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    note_type = Column(Enum(NoteType, name="note_type"), nullable=False)
    text = Column(Text, nullable=True)
    audio_data = Column(LargeBinary, nullable=True) 
    embedding = Column(Text, nullable=True) 
    created_at = Column(TIMESTAMP, server_default=func.now())

async def create_tables():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE user_preferences ADD COLUMN session_drafts TEXT DEFAULT '{}'"))
    except: pass

# ==========================================
# POMOCNICZE FUNKCJE BAZY DANYCH
# ==========================================
async def load_user_preferences(user_id):
    async_session = sessionmaker(bind=get_engine(), class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        res = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = res.scalars().first()
        if not prefs:
            current_lang = st.session_state.get('ui_language', 'Polski')
            prefs = UserPreferences(user_id=user_id, ui_language=current_lang)
            session.add(prefs)
            await session.commit()
        return prefs

async def update_user_preference(user_id, key, value):
    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
        res = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = res.scalars().first()
        if prefs:
            setattr(prefs, key, value)
            await session.commit()

async def save_note_to_db(text):
    emb_json = None
    if SEMANTIC_SEARCH_AVAILABLE and text:
        emb = embedder.encode(text).tolist()
        emb_json = json.dumps(emb)
    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
        session.add(Note(user_id=st.session_state.current_user_id, text=text, note_type=NoteType.TEXT, embedding=emb_json))
        await session.commit()

async def save_current_drafts(current_user):
    if not current_user or not st.session_state.get('prefs_loaded'): return
    
    drafts_to_save = st.session_state.p_state.copy()
    current_drafts_json = json.dumps(drafts_to_save)
    if current_drafts_json != st.session_state.get('last_saved_drafts'):
        await update_user_preference(current_user, "session_drafts", current_drafts_json)
        st.session_state.last_saved_drafts = current_drafts_json

def calculate_sm2(quality: int, repetition: int, easiness_factor: float, interval: int):
    if quality >= 3:
        if repetition == 0: interval = 1
        elif repetition == 1: interval = 6
        else: interval = int(round(interval * easiness_factor))
        repetition += 1
    else:
        repetition = 0
        interval = 1
    easiness_factor = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if easiness_factor < 1.3: easiness_factor = 1.3
    return repetition, easiness_factor, interval

def cosine_similarity(v1, v2):
    # Wymuszamy tablice numpy dla szybkości i bezbłędności obliczeń wektorowych
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def get_model_and_params():
    provider = st.session_state.ai_provider
    openai_key = get_secret("OPENAI_API_KEY")
    groq_key = get_secret("GROQ_API_KEY")
    
    if provider == "OpenAI (Płatne)":
        if openai_key and openai_key != "BRAK_KLUCZA":
            return "gpt-4o", None, openai_key
        else:
            # Automatyczny fallback na Groq z komunikatem
            # st.warning(_("Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq."))
            return "openai/openai/gpt-oss-20b", "https://api.groq.com/openai/v1", groq_key

    return "openai/openai/gpt-oss-20b", "https://api.groq.com/openai/v1", groq_key

def log_to_langfuse_api(name, model, input_data, output_data, in_tokens, out_tokens, user_id, custom_metadata=None):
    try:
        pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sec = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        host = host.rstrip("/")
        if not pub or not sec: return
        
        auth = base64.b64encode(f"{pub}:{sec}".encode()).decode()
        clean_model = model[7:] if model.startswith("openai/") else model
        
        try: in_t = int(in_tokens) if in_tokens else (len(str(input_data)) // 4)
        except: in_t = 0
        try: out_t = int(out_tokens) if out_tokens else (len(str(output_data)) // 4)
        except: out_t = 0
        
        # Nowy format używany przez Langfuse API do poprawnego zliczania tokenów
        body_data = {
            "id": str(uuid.uuid4()),
            "name": name,
            "model": clean_model,
            "input": input_data,
            "output": output_data,
            "usageDetails": {
                "input": in_t,
                "output": out_t,
                "total": in_t + out_t
            },
            "promptTokens": in_t,
            "completionTokens": out_t,
            "totalTokens": in_t + out_t,
            "userId": user_id
        }
        if custom_metadata:
            body_data["metadata"] = custom_metadata

        timestamp_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        payload = {
            "batch": [{
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "timestamp": timestamp_str,
                "body": body_data
            }]
        }
        
        res = requests.post(
            f"{host}/api/public/ingestion", 
            json=payload, 
            headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"}, 
            timeout=5
        )
        
        if res.status_code not in [200, 201, 207]:
            print(f"⚠️ LANGFUSE ODRZUCIŁ PACZKĘ [{res.status_code}]: {res.text}")
        else:
            print(f"✅ Langfuse przyjął trace: {name} (Model: {clean_model}, Tokens: {in_t}+{out_t})")
            
    except Exception as e:
        print(f"❌ Błąd HTTP Langfuse: {e}")


async def monitor_llm(system_prompt, user_prompt, task_type: TaskType, return_json=False):
    model_name, api_base, api_key = get_model_and_params()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        kwargs = {"response_format": {"type": "json_object"}} if return_json else {}
        
        # UNIWERSALNE WYWOŁANIE - LiteLLM przyjmie absolutnie każdy model w locie
        response = await acompletion(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            messages=messages,
            **kwargs
        )
        
        output = response.choices[0].message.content
        
        # Wyciągamy tokeny bezpośrednio z odpowiedzi modelu (jeśli dostawca je zwrócił) lub liczymy zapasowo
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        out_tokens = getattr(usage, "completion_tokens", None) if usage else None
        
        if not in_tokens or not out_tokens:
            try:
                in_tokens = litellm.token_counter(model=model_name, messages=messages)
                out_tokens = litellm.token_counter(model=model_name, text=output)
            except:
                in_tokens, out_tokens = len(str(messages)) // 4, len(str(output)) // 4

        await asyncio.to_thread(
            log_to_langfuse_api, task_type.value, model_name, messages, output, in_tokens, out_tokens, st.session_state.current_user_id, {"source": "monitor_llm", "task": task_type.value}
        )
        
        async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
            session.add(ActivityLog(user_id=st.session_state.current_user_id, task_type=task_type, input_text=user_prompt, output_text=output))
            await session.commit()
            
        return output
    except Exception as e:
        return f"{_('Błąd AI')}: {e}"


async def generate_word_data(word, language, target_language):
    model_name, api_base, api_key = get_model_and_params()
    client = instructor.from_litellm(acompletion)

    system_prompt = (
        f"Jesteś wybitnym lingwistą. Tworzysz fiszkę dla słowa w języku '{language}'. "
        f"Oryginalne zdania (pole 'original') MUSZĄ być w języku '{language}'. "
        f"Definicje oraz tłumaczenia zdań (pole 'translation') MUSZĄ być napisane w języku '{target_language}'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analizuj słowo: '{word}' (język: {language})."}
    ]
    
    try:
        kwargs = {"api_base": api_base} if api_base else {}
        flashcard = await client.chat.completions.create(
            model=model_name,
            api_key=api_key,
            response_model=FlashcardData,
            messages=messages,
            **kwargs
        )
        
        flashcard_dict = json.loads(flashcard.model_dump_json())
        
        try:
            in_tokens = litellm.token_counter(model=model_name, messages=messages)
            out_tokens = litellm.token_counter(model=model_name, text=json.dumps(flashcard_dict))
        except:
            in_tokens, out_tokens = len(str(messages)) // 4, len(str(flashcard_dict)) // 4
        
        await asyncio.to_thread(
            log_to_langfuse_api, "flashcard_generation", model_name, messages, flashcard_dict, in_tokens, out_tokens, st.session_state.current_user_id, {"word": word, "language": language, "target_language": target_language}
        )
        
        g = flashcard.grammar_info
        features = [f for f in [g.gender, g.number, g.case, g.person, g.tense, g.aspect, g.degree] if f]
        formatted_grammar = f"{g.part_of_speech} ({', '.join(features)})" if features else g.part_of_speech
        
        # Generowanie zdań do wyświetlenia (oryginał ➔ tłumaczenie)
        formatted_sentences = [f"{s.original} ➔ {s.translation}" for s in flashcard.sentences]
        
        async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
            session.add(ActivityLog(
                user_id=st.session_state.current_user_id, 
                task_type=TaskType.REVISION, 
                input_text=f"Analizuj: {word} w języku {language}", 
                output_text=flashcard.model_dump_json()
            ))
            await session.commit()
            
        return {
            "base_form": flashcard.base_form,
            "grammar_info": formatted_grammar,
            "definitions": flashcard.definitions,
            "forms": flashcard.forms,
            "sentences": formatted_sentences
        }
    except Exception as e:
        print(f"Błąd generatora fiszek: {e}")
        return {}
    
def get_example_sentences_api(word, language_name, target_language_name):
    try:
        from_lang_code = TATOEBA_LANG_MAP.get(language_name, "eng")
        dest_lang_code = TATOEBA_LANG_MAP.get(target_language_name, "pol")
        
        url = f"https://dev.tatoeba.org/pl/api_v0/search?from={from_lang_code}&query={word}&trans_filter=1&trans_to={dest_lang_code}"
        
        headers = {"User-Agent": "Lingilo-App/1.0"}
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            examples = []
            
            for result in data.get("results", []):
                src = result.get("text", "").strip()
                tgt = ""
                
                translations = result.get("translations", [])
                if translations and isinstance(translations, list):
                    for trans_group in translations:
                        if isinstance(trans_group, list) and len(trans_group) > 0:
                            tgt = trans_group[0].get("text", "").strip()
                            break
                
                if src and tgt:
                    examples.append(f"{src} ➔ {tgt}")
                    if len(examples) == 5: break
                        
            return examples
        else: return []
    except Exception as e:
        print(f"Błąd API Tatoeba: {e}")
        return []

def get_async_openai_client():
    api_key = get_secret("GROQ_API_KEY", "BRAK_KLUCZA")
    return openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

async def transcribe_audio(audio_bytes):
    client = get_async_openai_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    try:
        res = await client.audio.transcriptions.create(file=audio_file, model="whisper-large-v3")
        return res.text
    except Exception as e:
        return f"{_('Błąd transkrypcji')}: {e}"

# ==========================================
# WYSZUKIWANIE SEMANTYCZNE & SŁOWNIK
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedder():
    try: return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except ImportError: return None

embedder = load_embedder()
SEMANTIC_SEARCH_AVAILABLE = embedder is not None

TTS_LANG_MAP = {
    "Polski": "pl", "Angielski": "en", "Niemiecki": "de", 
    "Holenderski": "nl", "Francuski": "fr", "Hiszpański": "es", 
    "Włoski": "it", "Portugalski": "pt", "Rosyjski": "ru", 
    "Ukraiński": "uk", "Białoruski": "be", "Słowacki": "sk", "Czeski": "cs"
}

TATOEBA_LANG_MAP = {
    "Polski": "pol", "Angielski": "eng", "Niemiecki": "deu", 
    "Holenderski": "nld", "Francuski": "fra", "Hiszpański": "spa", 
    "Włoski": "ita", "Portugalski": "por", "Rosyjski": "rus", 
    "Ukraiński": "ukr", "Białoruski": "bel", "Słowacki": "slk", "Czeski": "ces"
}

SUPPORTED_LANGUAGES = [
    "Polski", "Angielski", "Niemiecki", "Holenderski", "Francuski", "Hiszpański", "Włoski", 
    "Portugalski", "Rosyjski", "Ukraiński", "Białoruski", "Słowacki", "Czeski"
]

NATIVE_LANGUAGE_NAMES = {
    "Polski": "Polski", "Angielski": "English", "Niemiecki": "Deutsch",
    "Holenderski": "Nederlands", "Francuski": "Français", "Hiszpański": "Español",
    "Włoski": "Italiano", "Portugalski": "Português", "Rosyjski": "Русский",
    "Ukraiński": "Українська", "Białoruski": "Беларуская", "Słowacki": "Slovenčina",
    "Czeski": "Čeština"
}

class GrammarFeatures(BaseModel):
    part_of_speech: str = Field(description="Część mowy")
    gender: Optional[str] = Field(None, description="Rodzaj gramatyczny")
    number: Optional[str] = Field(None, description="Liczba")
    case: Optional[str] = Field(None, description="Przypadek")
    person: Optional[str] = Field(None, description="Osoba")
    tense: Optional[str] = Field(None, description="Czas")
    aspect: Optional[str] = Field(None, description="Aspekt")
    degree: Optional[str] = Field(None, description="Stopień")

class ExampleSentence(BaseModel):
    original: str = Field(description="Oryginalne zdanie przykładowe z użyciem tego słowa (w języku obcym). MUSI być wypełnione.")
    translation: str = Field(description="Tłumaczenie tego zdania na język bazowy użytkownika. MUSI być wypełnione.")

class FlashcardData(BaseModel):
    base_form: str = Field(description="Forma podstawowa słowa")
    grammar_info: GrammarFeatures
    definitions: List[str] = Field(description="Krótkie definicje lub tłumaczenia (maksymalnie 3)")
    forms: List[str] = Field(description="Najważniejsze formy odmienione. Maksymalnie 3.")
    sentences: List[ExampleSentence] = Field(description="Przykładowe zdania z użyciem tego słowa (od 2 do 4 zdań)")

STATIC_TRANSLATIONS = {
    "Angielski": {"Polski": "Polish", "Angielski": "English", "Niemiecki": "German", "Holenderski": "Dutch", "Francuski": "French", "Hiszpański": "Spanish", "Włoski": "Italian", "Portugalski": "Portuguese", "Rosyjski": "Russian", "Ukraiński": "Ukrainian", "Białoruski": "Belarusian", "Słowacki": "Slovak", "Czeski": "Czech", "Ustawienia": "Settings", "Język interfejsu aplikacji:": "App interface language:", "Dostawca modelu AI:": "AI model provider:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Paid option is temporarily unavailable. Using free Groq model.", "Wybierz zadanie:": "Select task:", "Groq (Darmowe chmurowe)": "Groq (Free Cloud)", "OpenAI (Płatne)": "OpenAI (Paid)", "Tłumaczenie": "Translation", "Korekta tekstu": "Text correction", "Generowanie audio": "Audio generation", "Zapytanie": "Query", "Notatki": "Notes", "Powtórki": "Flashcards", "Historia i Zapisane": "History & Saved", "Wprowadź tekst do przetłumaczenia": "Enter text to translate", "Twój tekst:": "Your text:", "Język docelowy": "Target language", "Tłumacz": "Translate", "Trwa tłumaczenie...": "Translating...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "You are a translator. Return only the clean translation.", "Korekta błędów i redakcja": "Error correction and editing", "Tekst do poprawy:": "Text to correct:", "Popraw tekst": "Correct text", "Analiza błędów...": "Analyzing errors.", "Popraw błędy. Zwróć poprawiony tekst.": "Correct errors. Return corrected text.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Then explain the changes made in bullet points and briefly justify why the new version is better.", "Zamiana tekstu na mowę": "Text to speech", "Wpisz tekst:": "Enter text:", "Generuj wymowę": "Generate speech", "Przygotowywanie...": "Preparing...", "Zapisz to Audio do Historii": "Save Audio to History", "Wygenerowane TTS": "Generated TTS", "Zapisano audio w historii!": "Audio saved in history!", "Czat z ekspertem językowym": "Chat with language expert", "Zadaj pytanie:": "Ask a question:", "Zapytaj": "Ask", "Piszę odpowiedź...": "Writing response...", "Jesteś ekspertem językowym.": "You are a language expert.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Save speech and text with semantic search", "🎤 Nagraj notatkę głosową": "🎤 Record a voice note", "Opcjonalnie: Transkrybuj mowę": "Optional: Transcribe speech", "Rozpoznawanie mowy...": "Recognizing speech...", "Błąd transkrypcji": "Transcription error", "Treść notatki tekstowej (lub transkrypcji):": "Text note content (or transcription):", "Zapisz Notatkę do bazy": "Save Note to database", "Notatka jest pusta!": "Note is empty!", "Zapisywanie i generowanie wektorów...": "Saving and generating vectors...", "Notatka zapisana pomyślnie!": "Note saved successfully!", "Twój system nauki (Spaced Repetition)": "Your learning system (Spaced Repetition)", "Nauka (SM-2)": "Study (SM-2)", "Dodaj Fiszki": "Add Flashcards", "Edytuj / Usuń": "Edit / Delete", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Congratulations! No words for today.", "Do powtórzenia": "To review", "Język": "Language", "Baza": "Base", "Gramatyka": "Grammar", "Pokaż odpowiedź": "Show answer", "Definicje": "Definitions", "Odmiany": "Inflections", "Przykłady": "Examples", "Nie pamiętam": "Don't remember", "Z trudem": "Hard", "Dobrze": "Good", "Z łatwością": "Easy", "Kreator z analizą lingwistyczną AI": "Creator with AI linguistic analysis", "Wpisz słowo lub wyrażenie:": "Enter word or expression:", "Język słowa:": "Word language:", "Pobierz dane przez AI": "Fetch data via AI", "AI analizuje słowo jako API...": "AI is analyzing the word...", "Pobierz zdania z bazy Tatoeba": "Fetch sentences from Tatoeba", "Szukam autentycznych przykładów...": "Searching for authentic examples...", "API nie znalazło przykładów.": "API found no examples.", "Wprowadź ręcznie": "Enter manually", "Forma podstawowa": "Base form", "Info gramatyczne": "Grammar info", "Wybierz definicje do zapisania:": "Select definitions to save:", "Wybierz przykłady zdań:": "Select example sentences:", "Zapisz gotową fiszkę": "Save flashcard", "Fiszka dodana pomyślnie!": "Flashcard added successfully!", "Twoje własne uwagi (opcjonalnie):": "Your custom notes (optional):", "Twoje uwagi": "Your notes", "Język tłumaczenia (definicje):": "Translation language (definitions):", "Słowo": "Word", "💾 Zapisz zmiany": "💾 Save changes", "Audio (opcjonalnie):": "Audio (optional):", "Generuj wymowę (TTS)": "Generate pronunciation (TTS)", "🎤 Nagraj wymowę": "🎤 Record pronunciation", "Usuń przypisane audio": "Delete assigned audio", "Następna": "Next", "🗑 Usuń fiszkę": "🗑 Delete flashcard", "Centrum Danych i Historii": "Data and History Center", "Pełna Historia Aktywności": "Full Activity History", "Baza Notatek (Wyszukiwanie Semantyczne)": "Notes Base (Semantic Search)", "Brak historii aktywności.": "No activity history.", "Wejście": "Input", "Wyjście": "Output", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Semantic search inactive. Text search works.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Search notes (by text or meaning):", "Typ": "Type", "📝 Zapisz jako notatkę": "📝 Save as note", "📇 Przekaż do fiszek": "📇 Send to flashcards", "🔊 Przekaż do TTS": "🔊 Send to TTS", "Zapisano w notatkach!": "Saved to notes!", "Zalogowano pomyślnie": "Logged in successfully", "Wyloguj": "Log out", "Zaloguj się do Lingilo": "Log in to Lingilo", "Zaloguj przez Auth0": "Log in via Auth0", "Błąd komunikacji z Auth0": "Auth0 communication error", "Błąd AI": "AI Error", "Błąd TTS": "TTS Error", "Lingilo - Twój language helper": "Lingilo - Your language helper", "Własne definicje (każda w nowej linii - opcjonalnie):": "Custom definitions (each on a new line - optional):", "np. definicja 1\ndefinicja 2": "e.g. definition 1\ndefinition 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Custom examples (each on a new line - optional):", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "e.g. This is my first example.\nAnd this is my second example.", "TEXT": "TEXT", "AUDIO": "AUDIO", "GENERATED_AUDIO": "GENERATED AUDIO", "TRANSLATION": "TRANSLATION", "CORRECTION": "CORRECTION", "TEXT_TO_SPEECH": "TEXT TO SPEECH", "INQUIRY": "INQUIRY", "REVISION": "REVISION"},
    "Niemiecki": {"Polski": "Polnisch", "Angielski": "Englisch", "Niemiecki": "Deutsch", "Holenderski": "Niederländisch", "Francuski": "Französisch", "Hiszpański": "Spanisch", "Włoski": "Italienisch", "Portugalski": "Portugiesisch", "Rosyjski": "Russisch", "Ukraiński": "Ukrainisch", "Białoruski": "Weißrussisch", "Słowacki": "Slowakisch", "Czeski": "Tschechisch", "Ustawienia": "Einstellungen", "Język interfejsu aplikacji:": "App-Sprache:", "Dostawca modelu AI:": "KI-Modellanbieter:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Bezahlte Option nicht verfügbar.", "Wybierz zadanie:": "Aufgabe wählen:", "Groq (Darmowe chmurowe)": "Groq (Kostenlose Cloud)", "OpenAI (Płatne)": "OpenAI (Kostenpflichtig)", "Tłumaczenie": "Übersetzung", "Korekta tekstu": "Textkorrektur", "Generowanie audio": "Audio-Generierung", "Zapytanie": "Abfrage", "Notatki": "Notizen", "Powtórki": "Lernkarten", "Historia i Zapisane": "Verlauf & Gespeichert", "Wprowadź tekst do przetłumaczenia": "Text zum Übersetzen eingeben", "Twój tekst:": "Dein Text:", "Język docelowy": "Zielsprache", "Tłumacz": "Übersetzen", "Trwa tłumaczenie...": "Übersetzung läuft...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Du bist Übersetzer. Gib nur die reine Übersetzung zurück.", "Korekta błędów i redakcja": "Fehlerkorrektur", "Tekst do poprawy:": "Zu korrigierender Text:", "Popraw tekst": "Text korrigieren", "Analiza błędów...": "Fehler analysieren...", "Popraw błędy. Zwróć poprawiony tekst.": "Fehler korrigieren. Korrigierten Text zurückgeben.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Erkläre dann die Änderungen in Stichpunkten und begründe kurz, warum die neue Version besser ist.", "Zamiana tekstu na mowę": "Text-zu-Sprache", "Wpisz tekst:": "Text eingeben:", "Generuj wymowę": "Aussprache generieren", "Przygotowywanie...": "Vorbereiten...", "Zapisz to Audio do Historii": "Audio im Verlauf speichern", "Wygenerowane TTS": "Generiertes TTS", "Zapisano audio w historii!": "Audio im Verlauf gespeichert!", "Czat z ekspertem językowym": "Chat mit Sprachexperten", "Zadaj pytanie:": "Frage stellen:", "Zapytaj": "Fragen", "Piszę odpowiedź...": "Antwort wird geschrieben...", "Jesteś ekspertem językowym.": "Du bist ein Sprachexperte.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Sprache und Text mit semantischer Suche speichern", "🎤 Nagraj notatkę głosową": "🎤 Sprachnotiz aufnehmen", "Opcjonalnie: Transkrybuj mowę": "Optional: Sprache transkribieren", "Rozpoznawanie mowy...": "Spracherkennung...", "Błąd transkrypcji": "Transkriptionsfehler", "Treść notatki tekstowej (lub transkrypcji):": "Inhalt der Textnotiz:", "Zapisz Notatkę do bazy": "Notiz speichern", "Notatka jest pusta!": "Notiz ist leer!", "Zapisywanie i generowanie wektorów...": "Speichern und Vektoren generieren...", "Notatka zapisana pomyślnie!": "Notiz erfolgreich gespeichert!", "Twój system nauki (Spaced Repetition)": "Dein Lernsystem", "Nauka (SM-2)": "Lernen (SM-2)", "Dodaj Fiszki": "Lernkarten hinzufügen", "Edytuj / Usuń": "Bearbeiten / Löschen", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Glückwunsch! Keine Wörter für heute.", "Do powtórzenia": "Zu wiederholen", "Język": "Sprache", "Baza": "Basis", "Gramatyka": "Grammatik", "Pokaż odpowiedź": "Antwort anzeigen", "Definicje": "Definitionen", "Odmiany": "Beugungen", "Przykłady": "Beispiele", "Nie pamiętam": "Nicht erinnert", "Z trudem": "Schwer", "Dobrze": "Gut", "Z łatwością": "Einfach", "Kreator z analizą lingwistyczną AI": "Ersteller mit KI-Linguistikanalyse", "Wpisz słowo lub wyrażenie:": "Wort oder Ausdruck eingeben:", "Język słowa:": "Wortsprache:", "Pobierz dane przez AI": "Daten über KI abrufen", "AI analizuje słowo jako API...": "KI analysiert das Wort...", "Pobierz zdania z bazy Tatoeba": "Sätze von Tatoeba abrufen", "Szukam autentycznych przykładów...": "Suche nach Beispielen...", "API nie znalazło przykładów.": "API hat keine Beispiele gefunden.", "Wprowadź ręcznie": "Manuell eingeben", "Forma podstawowa": "Grundform", "Info gramatyczne": "Grammatik-Info", "Wybierz definicje do zapisania:": "Definitionen auswählen:", "Wybierz przykłady zdań:": "Beispielsätze auswählen:", "Zapisz gotową fiszkę": "Lernkarte speichern", "Fiszka dodana pomyślnie!": "Lernkarte erfolgreich hinzugefügt!", "Twoje własne uwagi (opcjonalnie):": "Deine Notizen (optional):", "Twoje uwagi": "Deine Notizen", "Język tłumaczenia (definicje):": "Übersetzungssprache (Definitionen):", "Słowo": "Wort", "💾 Zapisz zmiany": "💾 Änderungen speichern", "Audio (opcjonalnie):": "Audio (optional):", "Generuj wymowę (TTS)": "Aussprache generieren (TTS)", "🎤 Nagraj wymowę": "🎤 Aussprache aufnehmen", "Usuń przypisane audio": "Audio löschen", "Następna": "Nächste", "🗑 Usuń fiszkę": "🗑 Lernkarte löschen", "Centrum Danych i Historii": "Daten- und Verlaufszentrum", "Pełna Historia Aktywności": "Vollständiger Aktivitätsverlauf", "Baza Notatek (Wyszukiwanie Semantyczne)": "Notizdatenbank (Semantische Suche)", "Brak historii aktywności.": "Kein Aktivitätsverlauf.", "Wejście": "Eingabe", "Wyjście": "Ausgabe", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Semantische Suche inaktiv.", "Szukaj w notatkach (po tekście lub znaczeniu):": "In Notizen suchen:", "Typ": "Typ", "📝 Zapisz jako notatkę": "📝 Als Notiz speichern", "📇 Przekaż do fiszek": "📇 Zu Lernkarten", "🔊 Przekaż do TTS": "🔊 An TTS", "Zapisano w notatkach!": "In Notizen gespeichert!", "Zalogowano pomyślnie": "Erfolgreich eingeloggt", "Wyloguj": "Abmelden", "Zaloguj się do Lingilo": "Bei Lingilo anmelden", "Zaloguj przez Auth0": "Über Auth0 anmelden", "Błąd komunikacji z Auth0": "Auth0-Fehler", "Błąd AI": "KI-Fehler", "Błąd TTS": "TTS-Fehler", "Lingilo - Twój pomocnik językowy": "Lingilo - Dein Sprachhelfer", "Własne definicje (każda w nowej linii - opcjonalnie):": "Eigene Definitionen:", "np. definicja 1\ndefinicja 2": "z.B. Def 1\nDef 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Eigene Beispiele:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "z.B. Beispiel 1\nBeispiel 2", "TEXT": "TEXT", "AUDIO": "AUDIO", "GENERATED_AUDIO": "GENERIERTE AUDIO", "TRANSLATION": "ÜBERSETZUNG", "CORRECTION": "KORREKTUR", "TEXT_TO_SPEECH": "TEXT-ZU-SPRACHE", "INQUIRY": "ANFRAGE", "REVISION": "WIEDERHOLUNG"},
    "Francuski": {"Polski": "Polonais", "Angielski": "Anglais", "Niemiecki": "Allemand", "Holenderski": "Néerlandais", "Francuski": "Français", "Hiszpański": "Espagnol", "Włoski": "Italien", "Portugalski": "Portugais", "Rosyjski": "Russe", "Ukraiński": "Ukrainien", "Białoruski": "Biélorusse", "Słowacki": "Slovaque", "Czeski": "Tchèque", "Ustawienia": "Paramètres", "Język interfejsu aplikacji:": "Langue de l'interface :", "Dostawca modelu AI:": "Fournisseur d'IA :", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Option payante indisponible.", "Wybierz zadanie:": "Sélectionner une tâche :", "Groq (Darmowe chmurowe)": "Groq (Cloud Gratuit)", "OpenAI (Płatne)": "OpenAI (Payant)", "Tłumaczenie": "Traduction", "Korekta tekstu": "Correction de texte", "Generowanie audio": "Génération audio", "Zapytanie": "Requête", "Notatki": "Notes", "Powtórki": "Révisions", "Historia i Zapisane": "Historique & Enregistré", "Wprowadź tekst do przetłumaczenia": "Entrez le texte à traduire", "Twój tekst:": "Votre texte :", "Język docelowy": "Langue cible", "Tłumacz": "Traduire", "Trwa tłumaczenie...": "Traduction en cours...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Vous êtes traducteur. Retournez uniquement la traduction.", "Korekta błędów i redakcja": "Correction", "Tekst do poprawy:": "Texte à corriger :", "Popraw tekst": "Corriger le texte", "Analiza błędów...": "Analyse des erreurs...", "Popraw błędy. Zwróć poprawiony tekst.": "Corrigez les erreurs.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Expliquez ensuite les modifications en points.", "Zamiana tekstu na mowę": "Synthèse vocale", "Wpisz tekst:": "Entrez le texte :", "Generuj wymowę": "Générer la voix", "Przygotowywanie...": "Préparation...", "Zapisz to Audio do Historii": "Enregistrer l'audio", "Wygenerowane TTS": "TTS Généré", "Zapisano audio w historii!": "Audio enregistré !", "Czat z ekspertem językowym": "Discutez avec un expert", "Zadaj pytanie:": "Posez une question :", "Zapytaj": "Demander", "Piszę odpowiedź...": "Rédaction de la réponse...", "Jesteś ekspertem językowym.": "Vous êtes un expert linguistique.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Enregistrez avec recherche sémantique", "🎤 Nagraj notatkę głosową": "🎤 Enregistrer note vocale", "Opcjonalnie: Transkrybuj mowę": "Transcrire la voix", "Rozpoznawanie mowy...": "Reconnaissance vocale...", "Błąd transkrypcji": "Erreur de transcription", "Treść notatki tekstowej (lub transkrypcji):": "Contenu de la note :", "Zapisz Notatkę do bazy": "Enregistrer la note", "Notatka jest pusta!": "La note est vide !", "Zapisywanie i generowanie wektorów...": "Enregistrement en cours...", "Notatka zapisana pomyślnie!": "Note enregistrée !", "Twój system nauki (Spaced Repetition)": "Votre système d'apprentissage", "Nauka (SM-2)": "Étudier (SM-2)", "Dodaj Fiszki": "Ajouter Flashcards", "Edytuj / Usuń": "Modifier / Supprimer", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Félicitations ! Aucun mot aujourd'hui.", "Do powtórzenia": "À réviser", "Język": "Langue", "Baza": "Base", "Gramatyka": "Grammaire", "Pokaż odpowiedź": "Afficher la réponse", "Definicje": "Définitions", "Odmiany": "Déclinaisons", "Przykłady": "Exemples", "Nie pamiętam": "Oublié", "Z trudem": "Difficile", "Dobrze": "Bien", "Z łatwością": "Facile", "Kreator z analizą lingwistyczną AI": "Créateur avec analyse IA", "Wpisz słowo lub wyrażenie:": "Entrez un mot :", "Język słowa:": "Langue du mot :", "Pobierz dane przez AI": "Données IA", "AI analizuje słowo jako API...": "L'IA analyse le mot...", "Pobierz zdania z bazy Tatoeba": "Phrases de Tatoeba", "Szukam autentycznych przykładów...": "Recherche...", "API nie znalazło przykładów.": "L'API n'a rien trouvé.", "Wprowadź ręcznie": "Saisir manuellement", "Forma podstawowa": "Forme de base", "Info gramatyczne": "Infos grammaticales", "Wybierz definicje do zapisania:": "Sélectionnez les définitions :", "Wybierz przykłady zdań:": "Sélectionnez les phrases :", "Zapisz gotową fiszkę": "Enregistrer la flashcard", "Fiszka dodana pomyślnie!": "Flashcard ajoutée !", "Twoje własne uwagi (opcjonalnie):": "Vos notes (facultatif):", "Twoje uwagi": "Vos notes", "Język tłumaczenia (definicje):": "Langue de traduction :", "Słowo": "Mot", "💾 Zapisz zmiany": "💾 Enregistrer", "Audio (opcjonalnie):": "Audio (facultatif) :", "Generuj wymowę (TTS)": "Générer prononciation", "🎤 Nagraj wymowę": "🎤 Enregistrer prononciation", "Usuń przypisane audio": "Supprimer l'audio", "Następna": "Prochaine", "🗑 Usuń fiszkę": "🗑 Supprimer la flashcard", "Centrum Danych i Historii": "Centre de Données", "Pełna Historia Aktywności": "Historique complet", "Baza Notatek (Wyszukiwanie Semantyczne)": "Base de notes", "Brak historii aktywności.": "Aucun historique.", "Wejście": "Entrée", "Wyjście": "Sortie", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Recherche sémantique inactive.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Rechercher :", "Typ": "Type", "📝 Zapisz jako notatkę": "📝 Enregistrer comme note", "📇 Przekaż do fiszek": "📇 Vers flashcards", "🔊 Przekaż do TTS": "🔊 Vers TTS", "Zapisano w notatkach!": "Enregistré !", "Zalogowano pomyślnie": "Connexion réussie", "Wyloguj": "Se déconnecter", "Zaloguj się do Lingilo": "Connectez-vous à Lingilo", "Zaloguj przez Auth0": "Se connecter via Auth0", "Błąd komunikacji z Auth0": "Erreur Auth0", "Błąd AI": "Erreur IA", "Błąd TTS": "Erreur TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Votre assistant", "Własne definicje (każda w nowej linii - opcjonalnie):": "Définitions personnalisées :", "np. definicja 1\ndefinicja 2": "ex. déf 1\ndéf 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Exemples personnalisés :", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "ex. Exemple 1\nExemple 2", "TEXT": "TEXTE", "AUDIO": "AUDIO", "GENERATED_AUDIO": "AUDIO GÉNÉRÉ", "TRANSLATION": "TRADUCTION", "CORRECTION": "CORRECTION", "TEXT_TO_SPEECH": "SYNTHÈSE VOCALE", "INQUIRY": "REQUÊTE", "REVISION": "RÉVISION"},
    "Hiszpański": {"Polski": "Polaco", "Angielski": "Inglés", "Niemiecki": "Alemán", "Holenderski": "Holandés", "Francuski": "Francés", "Hiszpański": "Español", "Włoski": "Italiano", "Portugalski": "Portugués", "Rosyjski": "Ruso", "Ukraiński": "Ucraniano", "Białoruski": "Bielorruso", "Słowacki": "Eslovaco", "Czeski": "Checo", "Ustawienia": "Configuración", "Język interfejsu aplikacji:": "Idioma de la interfaz:", "Dostawca modelu AI:": "Proveedor de IA:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Opción de pago no disponible.", "Wybierz zadanie:": "Seleccionar tarea:", "Groq (Darmowe chmurowe)": "Groq (Nube Gratuita)", "OpenAI (Płatne)": "OpenAI (De pago)", "Tłumaczenie": "Traducción", "Korekta tekstu": "Corrección de texto", "Generowanie audio": "Generación de audio", "Zapytanie": "Consulta", "Notatki": "Notas", "Powtórki": "Tarjetas", "Historia i Zapisane": "Historial y Guardado", "Wprowadź tekst do przetłumaczenia": "Ingrese texto para traducir", "Twój tekst:": "Tu texto:", "Język docelowy": "Idioma de destino", "Tłumacz": "Traducir", "Trwa tłumaczenie...": "Traduciendo...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Eres un traductor. Devuelve solo la traducción pura.", "Korekta błędów i redakcja": "Corrección de errores", "Tekst do poprawy:": "Texto a corregir:", "Popraw tekst": "Corregir texto", "Analiza błędów...": "Analizando errores...", "Popraw błędy. Zwróć poprawiony tekst.": "Corrige errores.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Luego explica los cambios realizados.", "Zamiana tekstu na mowę": "Texto a voz", "Wpisz tekst:": "Ingrese texto:", "Generuj wymowę": "Generar voz", "Przygotowywanie...": "Preparando...", "Zapisz to Audio do Historii": "Guardar audio", "Wygenerowane TTS": "TTS Generado", "Zapisano audio w historii!": "¡Audio guardado!", "Czat z ekspertem językowym": "Chat con un experto", "Zadaj pytanie:": "Haz una pregunta:", "Zapytaj": "Preguntar", "Piszę odpowiedź...": "Escribiendo...", "Jesteś ekspertem językowym.": "Eres un experto en idiomas.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Guarda con búsqueda semántica", "🎤 Nagraj notatkę głosową": "🎤 Grabar nota de voz", "Opcjonalnie: Transkrybuj mowę": "Transcribir voz", "Rozpoznawanie mowy...": "Reconocimiento...", "Błąd transkrypcji": "Error de transcripción", "Treść notatki tekstowej (lub transkrypcji):": "Contenido de la nota:", "Zapisz Notatkę do bazy": "Guardar nota", "Notatka jest pusta!": "¡La nota está vacía!", "Zapisywanie i generowanie wektorów...": "Guardando...", "Notatka zapisana pomyślnie!": "¡Nota guardada!", "Twój system nauki (Spaced Repetition)": "Tu sistema de aprendizaje", "Nauka (SM-2)": "Estudiar (SM-2)", "Dodaj Fiszki": "Añadir Tarjetas", "Edytuj / Usuń": "Editar / Eliminar", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 ¡Felicidades! No hay palabras hoy.", "Do powtórzenia": "Para repasar", "Język": "Idioma", "Baza": "Base", "Gramatyka": "Gramática", "Pokaż odpowiedź": "Mostrar respuesta", "Definicje": "Definiciones", "Odmiany": "Inflexiones", "Przykłady": "Ejemplos", "Nie pamiętam": "No recuerdo", "Z trudem": "Difícil", "Dobrze": "Bien", "Z łatwością": "Fácil", "Kreator z analizą lingwistyczną AI": "Creador con análisis IA", "Wpisz słowo lub wyrażenie:": "Introduce una palabra:", "Język słowa:": "Idioma de la palabra:", "Pobierz dane przez AI": "Obtener datos por IA", "AI analizuje słowo jako API...": "La IA analiza la palabra...", "Pobierz zdania z bazy Tatoeba": "Obtener oraciones de Tatoeba", "Szukam autentycznych przykładów...": "Buscando...", "API nie znalazło przykładów.": "La API no encontró ejemplos.", "Wprowadź ręcznie": "Introducir manualmente", "Forma podstawowa": "Forma base", "Info gramatyczne": "Info gramatical", "Wybierz definicje do zapisania:": "Selecciona definiciones:", "Wybierz przykłady zdań:": "Selecciona oraciones:", "Zapisz gotową fiszkę": "Guardar tarjeta", "Fiszka dodana pomyślnie!": "¡Tarjeta añadida!", "Twoje własne uwagi (opcjonalnie):": "Tus notas (opcional):", "Twoje uwagi": "Tus notas", "Język tłumaczenia (definicje):": "Idioma de traducción:", "Słowo": "Palabra", "💾 Zapisz zmiany": "💾 Guardar", "Audio (opcjonalnie):": "Audio (opcional):", "Generuj wymowę (TTS)": "Generar pronunciación", "🎤 Nagraj wymowę": "🎤 Grabar pronunciación", "Usuń przypisane audio": "Eliminar audio", "Następna": "Próxima", "🗑 Usuń fiszkę": "🗑 Eliminar tarjeta", "Centrum Danych i Historii": "Centro de Datos", "Pełna Historia Aktywności": "Historial de actividad", "Baza Notatek (Wyszukiwanie Semantyczne)": "Base de Notas", "Brak historii aktywności.": "Sin historial.", "Wejście": "Entrada", "Wyjście": "Salida", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Búsqueda semántica inactiva.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Buscar en notas:", "Typ": "Tipo", "📝 Zapisz jako notatkę": "📝 Guardar nota", "📇 Przekaż do fiszek": "📇 A tarjetas", "🔊 Przekaż do TTS": "🔊 A TTS", "Zapisano w notatkach!": "¡Guardado!", "Zalogowano pomyślnie": "Inicio de sesión exitoso", "Wyloguj": "Cerrar sesión", "Zaloguj się do Lingilo": "Iniciar sesión en Lingilo", "Zaloguj przez Auth0": "Iniciar sesión a través de Auth0", "Błąd komunikacji z Auth0": "Error de Auth0", "Błąd AI": "Error de IA", "Błąd TTS": "Error de TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Tu asistente", "Własne definicje (każda w nowej linii - opcjonalnie):": "Definiciones propias:", "np. definicja 1\ndefinicja 2": "ej. definición 1\ndefinición 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Ejemplos propios:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "ej. Ejemplo 1\nEjemplo 2", "TEXT": "TEXTO", "AUDIO": "AUDIO", "GENERATED_AUDIO": "AUDIO GENERADO", "TRANSLATION": "TRADUCCIÓN", "CORRECTION": "CORRECCIÓN", "TEXT_TO_SPEECH": "TEXTO A VOZ", "INQUIRY": "CONSULTA", "REVISION": "REVISIÓN"},
    "Włoski": {"Polski": "Polacco", "Angielski": "Inglese", "Niemiecki": "Tedesco", "Holenderski": "Olandese", "Francuski": "Francese", "Hiszpański": "Spagnolo", "Włoski": "Italiano", "Portugalski": "Portoghese", "Rosyjski": "Russo", "Ukraiński": "Ucraino", "Białoruski": "Bielorusso", "Słowacki": "Slovacco", "Czeski": "Ceco", "Ustawienia": "Impostazioni", "Język interfejsu aplikacji:": "Lingua dell'interfaccia:", "Dostawca modelu AI:": "Fornitore AI:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Opzione non disponibile.", "Wybierz zadanie:": "Seleziona attività:", "Groq (Darmowe chmurowe)": "Groq (Cloud Gratuito)", "OpenAI (Płatne)": "OpenAI (A pagamento)", "Tłumaczenie": "Traduzione", "Korekta tekstu": "Correzione", "Generowanie audio": "Generazione audio", "Zapytanie": "Domanda", "Notatki": "Note", "Powtórki": "Ripassi", "Historia i Zapisane": "Cronologia", "Wprowadź tekst do przetłumaczenia": "Inserisci il testo", "Twój tekst:": "Il tuo testo:", "Język docelowy": "Lingua di destinazione", "Tłumacz": "Traduci", "Trwa tłumaczenie...": "Traduzione in corso...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Sei un traduttore. Restituisci solo la traduzione.", "Korekta błędów i redakcja": "Correzione errori", "Tekst do poprawy:": "Testo da correggere:", "Popraw tekst": "Correggi testo", "Analiza błędów...": "Analisi errori...", "Popraw błędy. Zwróć poprawiony tekst.": "Correggi gli errori.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Quindi spiega le modifiche.", "Zamiana tekstu na mowę": "Testo a voce", "Wpisz tekst:": "Inserisci testo:", "Generuj wymowę": "Genera voce", "Przygotowywanie...": "Preparazione...", "Zapisz to Audio do Historii": "Salva audio", "Wygenerowane TTS": "TTS Generato", "Zapisano audio w historii!": "Audio salvato!", "Czat z ekspertem językowym": "Chat con esperto", "Zadaj pytanie:": "Fai una domanda:", "Zapytaj": "Chiedi", "Piszę odpowiedź...": "Scrittura...", "Jesteś ekspertem językowym.": "Sei un esperto.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Salva con ricerca semantica", "🎤 Nagraj notatkę głosową": "🎤 Registra nota", "Opcjonalnie: Transkrybuj mowę": "Trascrivi voce", "Rozpoznawanie mowy...": "Riconoscimento...", "Błąd transkrypcji": "Errore trascrizione", "Treść notatki tekstowej (lub transkrypcji):": "Contenuto della nota:", "Zapisz Notatkę do bazy": "Salva nota", "Notatka jest pusta!": "La nota è vuota!", "Zapisywanie i generowanie wektorów...": "Salvataggio...", "Notatka zapisana pomyślnie!": "Nota salvata!", "Twój system nauki (Spaced Repetition)": "Il tuo sistema", "Nauka (SM-2)": "Studio (SM-2)", "Dodaj Fiszki": "Aggiungi Flashcard", "Edytuj / Usuń": "Modifica / Elimina", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Congratulazioni! Nessuna parola.", "Do powtórzenia": "Da ripassare", "Język": "Lingua", "Baza": "Base", "Gramatyka": "Grammatica", "Pokaż odpowiedź": "Mostra risposta", "Definizje": "Definizioni", "Odmiany": "Declinazioni", "Przykłady": "Esempi", "Nie pamiętam": "Non ricordo", "Z trudem": "Difficile", "Dobrze": "Bene", "Z łatwością": "Facile", "Kreator z analizą lingwistyczną AI": "Creatore AI", "Wpisz słowo lub wyrażenie:": "Inserisci una parola:", "Język słowa:": "Lingua della parola:", "Pobierz dane przez AI": "Ottieni dati IA", "AI analizuje słowo jako API...": "L'IA analizza...", "Pobierz zdania z bazy Tatoeba": "Ottieni frasi", "Szukam autentycznych przykładów...": "Ricerca...", "API nie znalazło przykładów.": "L'API non ha trovato esempi.", "Wprowadź ręcznie": "Inserisci manualmente", "Forma base": "Forma base", "Info gramatyczne": "Info grammaticali", "Wybierz definicje do zapisania:": "Seleziona definizioni:", "Wybierz przykłady zdań:": "Seleziona frasi:", "Zapisz gotową fiszkę": "Salva flashcard", "Fiszka dodana pomyślnie!": "Flashcard aggiunta!", "Twoje własne uwagi (opcjonalnie):": "I tuoi appunti (opzionale):", "Twoje uwagi": "I tuoi appunti", "Język tłumaczenia (definicje):": "Lingua di traduzione :", "Słowo": "Parola", "💾 Zapisz zmiany": "💾 Salva", "Audio (opcjonalnie):": "Audio (opzionale):", "Generuj wymowę (TTS)": "Genera pronuncia", "🎤 Nagraj wymowę": "🎤 Registra pronuncia", "Usuń przypisane audio": "Elimina audio", "Następna": "Prossima", "🗑 Usuń fiszkę": "🗑 Elimina flashcard", "Centrum Danych i Historii": "Centro Dati", "Pełna Historia Aktywności": "Cronologia", "Baza Notatek (Wyszukiwanie Semantyczne)": "Base Note", "Brak historii aktywności.": "Nessuna cronologia.", "Wejście": "Ingresso", "Wyjście": "Uscita", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Ricerca semantica disattiva.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Cerca:", "Typ": "Tipo", "📝 Zapisz jako notatkę": "📝 Salva nota", "📇 Przekaż do fiszek": "📇 Invia a flashcard", "🔊 Przekaż do TTS": "🔊 Invia a TTS", "Zapisano w notatkach!": "Salvato!", "Zalogowano pomyślnie": "Accesso effettuato", "Wyloguj": "Esci", "Zaloguj się do Lingilo": "Accedi a Lingilo", "Zaloguj przez Auth0": "Accedi tramite Auth0", "Błąd komunikacji z Auth0": "Errore Auth0", "Błąd AI": "Errore IA", "Błąd TTS": "Errore TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Il tuo assistente", "Własne definicje (każda w nowej linii - opcjonalnie):": "Definizioni personalizzate:", "np. definicja 1\ndefinicja 2": "es. definizione 1", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Esempi personalizzati:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "es. Esempio 1", "TEXT": "TESTO", "AUDIO": "AUDIO", "GENERATED_AUDIO": "AUDIO GENERATO", "TRANSLATION": "TRADUZIONE", "CORRECTION": "CORREZIONE", "TEXT_TO_SPEECH": "TESTO A VOCE", "INQUIRY": "DOMANDA", "REVISION": "RIPASSO"},
    "Holenderski": {"Polski": "Pools", "Angielski": "Engels", "Niemiecki": "Duits", "Holenderski": "Nederlands", "Francuski": "Frans", "Hiszpański": "Spaans", "Włoski": "Italiaans", "Portugalski": "Portugees", "Rosyjski": "Russisch", "Ukraiński": "Oekraïens", "Białoruski": "Wit-Russisch", "Słowacki": "Slowaaks", "Czeski": "Tsjechisch", "Ustawienia": "Instellingen", "Język interfejsu aplikacji:": "App interfacetaal:", "Dostawca modelu AI:": "AI aanbieder:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Betaalde optie niet beschikbaar.", "Wybierz zadanie:": "Selecteer taak:", "Groq (Darmowe chmurowe)": "Groq (Gratis Cloud)", "OpenAI (Płatne)": "OpenAI (Betaald)", "Tłumaczenie": "Vertaling", "Korekta tekstu": "Tekstcorrectie", "Generowanie audio": "Audio-generatie", "Zapytanie": "Vraag", "Notatki": "Notities", "Powtórki": "Flashcards", "Historia i Zapisane": "Geschiedenis & Opgeslagen", "Wprowadź tekst do przetłumaczenia": "Voer tekst in om te vertalen", "Twój tekst:": "Jouw tekst:", "Język docelowy": "Doeltaal", "Tłumacz": "Vertaal", "Trwa tłumaczenie...": "Vertalen...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Je bent een vertaler.", "Korekta błędów i redakcja": "Foutcorrectie", "Tekst do poprawy:": "Tekst om te corrigeren:", "Popraw tekst": "Corrigeer tekst", "Analiza błędów...": "Fouten analyseren...", "Popraw błędy. Zwróć poprawiony tekst.": "Corrigeer fouten.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Leg vervolgens de wijzigingen uit.", "Zamiana tekstu na mowę": "Tekst naar spraak", "Wpisz tekst:": "Voer tekst in:", "Generuj wymowę": "Genereer spraak", "Przygotowywanie...": "Voorbereiden...", "Zapisz to Audio do Historii": "Sla audio op", "Wygenerowane TTS": "Gegenereerde TTS", "Zapisano audio w historii!": "Audio opgeslagen!", "Czat z ekspertem językowym": "Chat met expert", "Zadaj pytanie:": "Stel een vraag:", "Zapytaj": "Vraag", "Piszę odpowiedź...": "Antwoord schrijven...", "Jesteś ekspertem językowym.": "Je bent een taalexpert.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Sla op met semantisch zoeken", "🎤 Nagraj notatkę głosową": "🎤 Neem spraaknotitie op", "Opcjonalnie: Transkrybuj mowę": "Transcribeer spraak", "Rozpoznawanie mowy...": "Spraakherkenning...", "Błąd transkrypcji": "Transcriptiefout", "Treść notatki tekstowej (lub transkrypcji):": "Inhoud van notitie:", "Zapisz Notatkę do bazy": "Sla notitie op", "Notatka jest pusta!": "Notitie is leeg!", "Zapisywanie i generowanie wektorów...": "Opslaan...", "Notatka zapisana pomyślnie!": "Notitie succesvol opgeslagen!", "Twój system nauki (Spaced Repetition)": "Jouw leersysteem", "Nauka (SM-2)": "Studeren (SM-2)", "Dodaj Fiszki": "Voeg Flashcards toe", "Edytuj / Usuń": "Bewerken / Verwijderen", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Gefeliciteerd! Geen woorden.", "Do powtórzenia": "Te herhalen", "Język": "Taal", "Baza": "Basis", "Gramatyka": "Grammatica", "Pokaż odpowiedź": "Toon antwoord", "Definicje": "Definities", "Odmiany": "Verbuigingen", "Przykłady": "Voorbeelden", "Nie pamiętam": "Vergeten", "Z trudem": "Moeilijk", "Dobrze": "Goed", "Z łatwością": "Makkelijk", "Kreator z analizą lingwistyczną AI": "Maker met AI-analyse", "Wpisz słowo lub wyrażenie:": "Voer een woord in:", "Język słowa:": "Taal van het woord:", "Pobierz dane przez AI": "Haal gegevens op (AI)", "AI analizuje słowo jako API...": "AI analyseert...", "Pobierz zdania z bazy Tatoeba": "Haal zinnen op", "Szukam autentycznych przykładów...": "Zoeken...", "API nie znalazło przykładów.": "API heeft niets gevonden.", "Wprowadź ręcznie": "Handmatig invoeren", "Forma podstawowa": "Basisvorm", "Info gramatyczne": "Grammatica-info", "Wybierz definicje do zapisania:": "Selecteer definities:", "Wybierz przykłady zdań:": "Selecteer zinnen:", "Zapisz gotową fiszkę": "Sla flashcard op", "Fiszka dodana pomyślnie!": "Flashcard toegevoegd!", "Twoje własne uwagi (opcjonalnie):": "Jouw eigen notities:", "Twoje uwagi": "Jouw notities", "Język tłumaczenia (definicje):": "Vertaaltaal:", "Słowo": "Woord", "💾 Zapisz zmiany": "💾 Opslaan", "Audio (opcjonalnie):": "Audio (optioneel):", "Generuj wymowę (TTS)": "Genereer uitspraak", "🎤 Nagraj wymowę": "🎤 Neem uitspraak op", "Usuń przypisane audio": "Audio verwijderen", "Następna": "Volgende", "🗑 Usuń fiszkę": "🗑 Verwijder flashcard", "Centrum Danych i Historii": "Datacentrum", "Pełna Historia Aktywności": "Geschiedenis", "Baza Notatek (Wyszukiwanie Semantyczne)": "Notitiebasis", "Brak historii aktywności.": "Geen geschiedenis.", "Wejście": "Invoer", "Wyjście": "Uitvoer", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Semantisch zoeken inactief.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Zoek:", "Typ": "Type", "📝 Zapisz jako notatkę": "📝 Opslaan als notitie", "📇 Przekaż do fiszek": "📇 Naar flashcards", "🔊 Przekaż do TTS": "🔊 Naar TTS", "Zapisano w notatkach!": "Opgeslagen!", "Zalogowano pomyślnie": "Succesvol ingelogd", "Wyloguj": "Uitloggen", "Zaloguj się do Lingilo": "Log in bij Lingilo", "Zaloguj door Auth0": "Inloggen via Auth0", "Błąd komunikacji z Auth0": "Communicatiefout Auth0", "Błąd AI": "AI-fout", "Błąd TTS": "TTS-fout", "Lingilo - Twój pomocnik językowy": "Lingilo - Jouw assistent", "Własne definicje (każda w nowej linii - opcjonalnie):": "Eigen definities:", "np. definicja 1\ndefinicja 2": "bijv. def 1\ndef 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Eigen voorbeelden:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "bijv. Voorbeeld 1\nVoorbeeld 2", "TEXT": "TEKST", "AUDIO": "AUDIO", "GENERATED_AUDIO": "GEGENEREERDE AUDIO", "TRANSLATION": "VERTALING", "CORRECTION": "CORRECTIE", "TEXT_TO_SPEECH": "TEKST NAAR SPRAAK", "INQUIRY": "VRAAG", "REVISION": "HERZIENING"},
    "Czeski": {"Polski": "Polština", "Angielski": "Angličtina", "Niemiecki": "Němčina", "Holenderski": "Nizozemština", "Francuski": "Francouzština", "Hiszpański": "Španělština", "Włoski": "Italština", "Portugalski": "Portugalština", "Rosyjski": "Ruština", "Ukraiński": "Ukrajinština", "Białoruski": "Běloruština", "Słowacki": "Slovenština", "Czeski": "Čeština", "Ustawienia": "Nastavení", "Język interfejsu aplikacji:": "Jazyk rozhraní:", "Dostawca modelu AI:": "Poskytovatel AI:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Placená verze není dostupná.", "Wybierz zadanie:": "Vyberte úkol:", "Groq (Darmowe chmurowe)": "Groq (Bezplatný Cloud)", "OpenAI (Płatne)": "OpenAI (Placené)", "Tłumaczenie": "Překlad", "Korekta tekstu": "Oprava textu", "Generowanie audio": "Generování zvuku", "Zapytanie": "Dotaz", "Notatki": "Poznámky", "Powtórki": "Kartičky", "Historia i Zapisane": "Historie", "Wprowadź tekst do przetłumaczenia": "Zadejte text k překladu", "Twój tekst:": "Váš text:", "Język docelowy": "Cílový jazyk", "Tłumacz": "Přeložit", "Trwa tłumaczenie...": "Překládám...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Jste překladatel. Vraťte pouze překlad.", "Korekta błędów i redakcja": "Oprava chyb", "Tekst do poprawy:": "Text k opravě:", "Popraw tekst": "Opravit text", "Analiza błędów...": "Analýza chyb...", "Popraw błędy. Zwróć poprawiony tekst.": "Opravte chyby. Vraťte opravený text.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Poté v bodech vysvětlete provedené změny.", "Zamiana tekstu na mowę": "Text na řeč", "Wpisz tekst:": "Zadejte text:", "Generuj wymowę": "Generovat výslovnost", "Przygotowywanie...": "Připravuji...", "Zapisz to Audio do Historii": "Uložit zvuk", "Wygenerowane TTS": "Generované TTS", "Zapisano audio w historii!": "Zvuk uložen!", "Czat z ekspertem językowym": "Chat s expertem", "Zadaj pytanie:": "Položte otázku:", "Zapytaj": "Zeptat se", "Piszę odpowiedź...": "Píšu odpověď...", "Jesteś ekspertem językowym.": "Jste jazykový expert.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Uložte řeč s sémantickým vyhledáváním", "🎤 Nagraj notatkę głosową": "🎤 Nahrát poznámku", "Opcjonalnie: Transkrybuj mowę": "Přepsat řeč", "Rozpoznawanie mowy...": "Rozpoznávanie řeči...", "Błąd transkrypcji": "Chyba přepisu", "Treść notatki tekstowej (lub transkrypcji):": "Obsah poznámky:", "Zapisz Notatkę do bazy": "Uložit poznámku", "Notatka jest pusta!": "Poznámka je prázdná!", "Zapisywanie i generowanie wektorów...": "Ukládám...", "Notatka zapisana pomyślnie!": "Poznámka uložena!", "Twój system nauki (Spaced Repetition)": "Váš výukový systém", "Nauka (SM-2)": "Studium (SM-2)", "Dodaj Fiszki": "Přidat kartičky", "Edytuj / Usuń": "Upravit / Smazat", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Gratulujeme! Dnes žádná slovíčka.", "Do powtórzenia": "K opakování", "Język": "Jazyk", "Baza": "Základ", "Gramatyka": "Gramatika", "Pokaż odpowiedź": "Ukázat odpověď", "Definicje": "Definice", "Odmiany": "Skloňování", "Przykłady": "Příklady", "Nie pamiętam": "Nepamatuji", "Z trudem": "Těžké", "Dobrze": "Dobře", "Z łatwością": "Snadné", "Kreator z analizą lingwistyczną AI": "Tvůrce s AI analýzou", "Wpisz słowo lub wyrażenie:": "Zadejte slovo:", "Język słowa:": "Jazyk slova:", "Pobierz dane przez AI": "Získat data z AI", "AI analizuje słowo jako API...": "AI analyzuje slovo...", "Pobierz zdania z bazy Tatoeba": "Získat věty z Tatoeba", "Szukam autentycznych przykładów...": "Hledám příklady...", "API nie znalazło příkladů.": "API nenašlo příklady.", "Wprowadź ręcznie": "Zadat ručně", "Forma podstawowa": "Základní forma", "Info gramatyczne": "Gramatické info", "Wybierz definicje do zapisania:": "Vyberte definice:", "Wybierz przykłady zdań:": "Vyberte věty:", "Zapisz gotową fiszkę": "Uložit kartičku", "Fiszka dodana pomyślnie!": "Kartička přidána!", "Twoje własne uwagi (opcjonalnie):": "Vaše vlastní poznámky:", "Twoje uwagi": "Vaše poznámky", "Język tłumaczenia (definicje):": "Jazyk překladu:", "Słowo": "Slovo", "💾 Zapisz zmiany": "💾 Uložit změny", "Audio (opcjonalnie):": "Zvuk (volitelné):", "Generuj wymowę (TTS)": "Generovat výslovnost", "🎤 Nagraj wymowę": "🎤 Nahrát výslovnost", "Usuń przypisane audio": "Odstranit zvuk", "Następna": "Další", "🗑 Usuń fiszkę": "🗑 Smazat kartičku", "Centrum Danych i Historii": "Datové centrum", "Pełna Historia Aktywności": "Kompletní historie", "Baza Notatek (Wyszukiwanie Semantyczne)": "Poznámky", "Brak historii aktywności.": "Žádná historie.", "Wejście": "Vstup", "Wyjście": "Výstup", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Sémantické hledání neaktivní.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Hledat:", "Typ": "Typ", "📝 Zapisz jako notatkę": "📝 Uložit jako poznámku", "📇 Przekaż do fiszek": "📇 Odeslat do kartiček", "🔊 Przekaż do TTS": "🔊 Odeslat do TTS", "Zapisano w notatkach!": "Uloženo v poznámkách!", "Zalogowano pomyślnie": "Úspěšně přihlášeno", "Wyloguj": "Odhlásit se", "Zaloguj się do Lingilo": "Přihlaste se do Lingilo", "Zaloguj przez Auth0": "Přihlásit přes Auth0", "Błąd komunikacji z Auth0": "Chyba komunikace s Auth0", "Błąd AI": "Chyba AI", "Błąd TTS": "Chyba TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Váš asistent", "Własne definicje (każda w nowej linii - opcjonalnie):": "Vlastní definice:", "np. definicja 1\ndefinicja 2": "např. definice 1\ndefinice 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Vlastní příklady:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "např. Příklad 1\nPříklad 2", "TEXT": "TEXT", "AUDIO": "AUDIO", "GENERATED_AUDIO": "GENEROVANÉ AUDIO", "TRANSLATION": "PŘEKLAD", "CORRECTION": "OPRAVA", "TEXT_TO_SPEECH": "TEXT NA ŘEČ", "INQUIRY": "DOTAZ", "REVISION": "OPAKOVÁNÍ"},
    "Portugalski": {"Polski": "Polonês", "Angielski": "Inglês", "Niemiecki": "Alemão", "Holenderski": "Holandês", "Francuski": "Francês", "Hiszpański": "Espanhol", "Włoski": "Italiano", "Portugalski": "Português", "Rosyjski": "Russo", "Ukraiński": "Ucraniano", "Białoruski": "Bielorrusso", "Słowacki": "Eslovaco", "Czeski": "Tcheco", "Ustawienia": "Configurações", "Język interfejsu aplikacji:": "Idioma da interface:", "Dostawca modelu AI:": "Provedor de IA:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "A opção paga está indisponível.", "Wybierz zadanie:": "Selecionar tarefa:", "Groq (Darmowe chmurowe)": "Groq (Nuvem Gratuita)", "OpenAI (Płatne)": "OpenAI (Pago)", "Tłumaczenie": "Tradução", "Korekta tekstu": "Correção de texto", "Generowanie audio": "Geração de áudio", "Zapytanie": "Consulta", "Notatki": "Notas", "Powtórki": "Revisões", "Historia i Zapisane": "Histórico e Salvos", "Wprowadź tekst do przetłumaczenia": "Insira o texto para traduzir", "Twój tekst:": "Seu texto:", "Język docelowy": "Idioma de destino", "Tłumacz": "Traduzir", "Trwa tłumaczenie...": "Traduzindo...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Você é um tradutor. Retorne apenas a tradução pura.", "Korekta błędów i redakcja": "Correção e edição de erros", "Tekst do poprawy:": "Texto a corrigir:", "Popraw tekst": "Corrigir texto", "Analiza błędów...": "Analisando erros...", "Popraw błędy. Zwróć poprawiony tekst.": "Corrija os erros.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Em seguida, explique as alterações.", "Zamiana tekstu na mowę": "Texto para fala", "Wpisz tekst:": "Digite o texto:", "Generuj wymowę": "Gerar pronúncia", "Przygotowywanie...": "Preparando...", "Zapisz to Audio do Historii": "Salvar Áudio no Histórico", "Wygenerowane TTS": "TTS Gerado", "Zapisano audio w historii!": "Áudio salvo no histórico!", "Czat z ekspertem językowym": "Chat com especialista", "Zadaj pytanie:": "Faça uma pergunta:", "Zapytaj": "Perguntar", "Piszę odpowiedź...": "Escrevendo resposta...", "Jesteś ekspertem językowym.": "Você é um especialista em idiomas.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Salve voz e texto com pesquisa semântica", "🎤 Nagraj notatkę głosową": "🎤 Gravar nota de voz", "Opcjonalnie: Transkrybuj mowę": "Transcrever voz", "Rozpoznawanie mowy...": "Reconhecimento de voz...", "Błąd transkrypcji": "Erro de transcrição", "Treść notatki tekstowej (lub transkrypcji):": "Conteúdo da nota:", "Zapisz Notatkę do bazy": "Salvar Nota", "Notatka jest pusta!": "A nota está vazia!", "Zapisywanie i generowanie wektorów...": "Salvando...", "Notatka zapisana pomyślnie!": "Nota salva com sucesso!", "Twój system nauki (Spaced Repetition)": "Seu sistema de aprendizado", "Nauka (SM-2)": "Estudo (SM-2)", "Dodaj Fiszki": "Adicionar Cartões", "Edytuj / Usuń": "Editar / Excluir", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Parabéns! Sem palavras por hoje.", "Do powtórzenia": "Para revisar", "Język": "Idioma", "Baza": "Base", "Gramatyka": "Gramática", "Pokaż odpowiedź": "Mostrar resposta", "Definicje": "Definições", "Odmiany": "Flexões", "Przykłady": "Exemplos", "Nie pamiętam": "Não lembro", "Z trudem": "Com dificuldade", "Dobrze": "Bom", "Z łatwością": "Fácil", "Kreator z analizą lingwistyczną AI": "Criador com análise de IA", "Wpisz słowo lub wyrażenie:": "Digite a palavra ou expressão:", "Język słowa:": "Idioma da palavra:", "Pobierz dane przez AI": "Buscar dados por IA", "AI analizuje słowo jako API...": "A IA está analisando a palavra...", "Pobierz zdania z bazy Tatoeba": "Buscar frases de Tatoeba", "Szukam autentycznych przykładów...": "Buscando exemplos...", "API nie znalazło przykładów.": "A API não encontrou exemplos.", "Wprowadź ręcznie": "Inserir manualmente", "Forma podstawowa": "Forma básica", "Info gramatyczne": "Info gramatical", "Wybierz definicje do zapisania:": "Selecione as definições:", "Wybierz przykłady zdań:": "Selecione frases:", "Zapisz gotową fiszkę": "Salvar cartão", "Fiszka dodana pomyślnie!": "Cartão adicionado!", "Twoje własne uwagi (opcjonalnie):": "Suas próprias notas (opcional):", "Twoje uwagi": "Suas notas", "Język tłumaczenia (definicje):": "Idioma de tradução (definições):", "Słowo": "Palavra", "💾 Zapisz zmiany": "💾 Salvar alterações", "Audio (opcjonalnie):": "Áudio (opcional):", "Generuj wymowę (TTS)": "Gerar pronúncia (TTS)", "🎤 Nagraj wymowę": "🎤 Gravar pronúncia", "Usuń przypisane audio": "Excluir áudio", "Następna": "Próxima", "🗑 Usuń fiszkę": "🗑 Excluir cartão", "Centrum Danych i Historii": "Centro de Dados", "Pełna Historia Aktywności": "Histórico Completo", "Baza Notatek (Wyszukiwanie Semantyczne)": "Base de Notas", "Brak historii aktywności.": "Sem histórico.", "Wejście": "Entrada", "Wyjście": "Saída", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Pesquisa semântica inativa.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Pesquisar nas notas:", "Typ": "Tipo", "📝 Zapisz jako notatkę": "📝 Salvar como nota", "📇 Przekaż do fiszek": "📇 Enviar para cartões", "🔊 Przekaż do TTS": "🔊 Enviar para TTS", "Zapisano w notatkach!": "Salvo nas notas!", "Zalogowano pomyślnie": "Login bem-sucedido", "Wyloguj": "Sair", "Zaloguj się do Lingilo": "Entrar no Lingilo", "Zaloguj przez Auth0": "Entrar via Auth0", "Błąd komunikacji z Auth0": "Erro de comunicação com Auth0", "Błąd AI": "Erro de IA", "Błąd TTS": "Erro de TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Seu assistente", "Własne definicje (każda w nowej linii - opcjonalnie):": "Definições próprias (cada uma em uma nova linha - opcional):", "np. definicja 1\ndefinicja 2": "ex. definição 1\ndefinição 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Exemplos próprios (cada um em uma nova linha - opcional):", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "ex. Exemplo 1\nExemplo 2", "TEXT": "TEXTO", "AUDIO": "ÁUDIO", "GENERATED_AUDIO": "ÁUDIO GERADO", "TRANSLATION": "TRADUÇÃO", "CORRECTION": "CORREÇÃO", "TEXT_TO_SPEECH": "TEXTO PARA FALA", "INQUIRY": "CONSULTA", "REVISION": "REVISÃO"},
    "Rosyjski": {"Polski": "Польский", "Angielski": "Английский", "Niemiecki": "Немецкий", "Holenderski": "Нидерландский", "Francuski": "Французский", "Hiszpański": "Испанский", "Włoski": "Итальянский", "Portugalski": "Португальский", "Rosyjski": "Русский", "Ukraiński": "Украинский", "Białoruski": "Белорусский", "Słowacki": "Словацкий", "Czeski": "Чешский", "Ustawienia": "Настройки", "Język interfejsu aplikacji:": "Язык интерфейса:", "Dostawca modelu AI:": "Провайдер ИИ:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Платная опция недоступна.", "Wybierz zadanie:": "Выберите задачу:", "Groq (Darmowe chmurowe)": "Groq (Бесплатное облако)", "OpenAI (Płatne)": "OpenAI (Платный)", "Tłumaczenie": "Перевод", "Korekta tekstu": "Исправление текста", "Generowanie audio": "Генерация аудио", "Zapytanie": "Запрос", "Notatki": "Заметки", "Powtórki": "Карточки", "Historia i Zapisane": "История и Сохраненное", "Wprowadź tekst do przetłumaczenia": "Введите текст для перевода", "Twój tekst:": "Ваш текст:", "Język docelowy": "Целевой язык", "Tłumacz": "Перевести", "Trwa tłumaczenie...": "Перевод...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Вы переводчик. Верните только чистый перевод.", "Korekta błędów i redakcja": "Исправление ошибок", "Tekst do poprawy:": "Текст для исправления:", "Popraw tekst": "Исправить текст", "Analiza błędów...": "Анализ ошибок...", "Popraw błędy. Zwróć poprawiony tekst.": "Исправьте ошибки.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Затем по пунктах объясните изменения.", "Zamiana tekstu na mowę": "Текст в речь", "Wpisz tekst:": "Введите текст:", "Generuj wymowę": "Сгенерировать произношение", "Przygotowywanie...": "Подготовка...", "Zapisz to Audio do Historii": "Сохранить аудио в историю", "Wygenerowane TTS": "Сгенерированное TTS", "Zapisano audio w historii!": "Аудио сохранено в истории!", "Czat z ekspertem językowym": "Чат с языковым экспертом", "Zadaj pytanie:": "Задайте вопрос:", "Zapytaj": "Спросить", "Piszę odpowiedź...": "Пишу ответ...", "Jesteś ekspertem językowym.": "Вы языковой эксперт.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Сохраняйте речь и текст с семантическим поиском", "🎤 Nagraj notatkę głosową": "🎤 Записать голосовую заметку", "Opcjonalnie: Transkrybuj mowę": "Дополнительно: Транскрибировать речь", "Rozpoznawanie mowy...": "Распознавание речи...", "Błąd transkrypcji": "Ошибка транскрипции", "Treść notatki tekstowej (lub transkrypcji):": "Текст заметки:", "Zapisz Notatkę do bazy": "Сохранить заметку", "Notatka jest pusta!": "Заметка пуста!", "Zapisywanie i generowanie wektorów...": "Сохранение...", "Notatka zapisana pomyślnie!": "Заметка успешно сохралена!", "Twój system nauki (Spaced Repetition)": "Ваша система обучения", "Nauka (SM-2)": "Обучение (SM-2)", "Dodaj Fiszki": "Добавить карточки", "Edytuj / Usuń": "Редактировать / Удалить", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Поздравляем! На сегодня слов нет.", "Do powtórzenia": "Для повторения", "Język": "Язык", "Baza": "База", "Gramatyka": "Грамматика", "Pokaż odpowiedź": "Показать ответ", "Definicje": "Определения", "Odmiany": "Склонения", "Przykłady": "Примеры", "Nie pamiętam": "Не помню", "Z trudem": "С трудом", "Dobrze": "Хорошо", "Z łatwością": "Легко", "Kreator z analizą lingwistyczną AI": "Создатель с анализом ИИ", "Wpisz słowo lub wyrażenie:": "Введите слово:", "Język słowa:": "Язык слова:", "Pobierz dane przez AI": "Получить данные через ИИ", "AI analizuje słowo jako API...": "ИИ анализирует слово...", "Pobierz zdania z bazy Tatoeba": "Получить предложения Tatoeba", "Szukam autentycznych przykładów...": "Поиск примеров...", "API nie znalazło przykładów.": "API не нашло примеров.", "Wprowadź ręcznie": "Ввести вручную", "Forma podstawowa": "Начальная форма", "Info gramatyczne": "Грамматическая инфо", "Wybierz definicje do zapisania:": "Выберите определения:", "Wybierz przykłady zdań:": "Выберите примеры:", "Zapisz gotową fiszkę": "Сохранить карточку", "Fiszka dodana pomyślnie!": "Карточка добавлена!", "Twoje własne uwagi (opcjonalnie):": "Ваши собственные заметки:", "Twoje uwagi": "Ваши заметки", "Język tłumaczenia (definicje):": "Язык перевода (определения):", "Słowo": "Слово", "💾 Zapisz zmiany": "💾 Сохранить изменения", "Audio (opcjonalnie):": "Аудио (необязательно):", "Generuj wymowę (TTS)": "Сгенерировать произношение (TTS)", "🎤 Nagraj wymowę": "🎤 Записать произношение", "Usuń przypisane audio": "Удалить назначенное аудио", "Następna": "Следующая", "🗑 Usuń fiszkę": "🗑 Удалить карточку", "Centrum Danych i Historii": "Центр данных и истории", "Pełna Historia Aktywności": "Полная история", "Baza Notatek (Wyszukiwanie Semantyczne)": "База заметок (Семантический поиск)", "Brak historii aktywności.": "Нет истории активности.", "Wejście": "Вход", "Wyjście": "Выход", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Семантический поиск неактивен.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Искать в заметках:", "Typ": "Тип", "📝 Zapisz jako notatkę": "📝 Сохранить как заметку", "📇 Przekaż do fiszek": "📇 Отправить в карточки", "🔊 Przekaż do TTS": "🔊 Отправить в TTS", "Zapisano w notatkach!": "Сохранено в заметках!", "Zalogowano pomyślnie": "Успешный вход", "Wyloguj": "Выйти", "Zaloguj się do Lingilo": "Войти в Lingilo", "Zaloguj przez Auth0": "Войти через Auth0", "Błąd komunikacji z Auth0": "Ошибка связи с Auth0", "Błąd AI": "Ошибка ИИ", "Błąd TTS": "Ошибка TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Ваш помощник", "Własne definicje (każda w nowej linii - opcjonalnie):": "Свои определения:", "np. definicja 1\ndefinicja 2": "напр. опред 1\nопред 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Свои примеры:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "напр. Пример 1\nПример 2", "TEXT": "ТЕКСТ", "AUDIO": "АУДИО", "GENERATED_AUDIO": "СГЕНЕРАВАНАЕ АУДИО", "TRANSLATION": "ПЕРЕВОД", "CORRECTION": "ИСПРАВЛЕНИЕ", "TEXT_TO_SPEECH": "ТЕКСТ В РЕЧЬ", "INQUIRY": "ЗАПРОС", "REVISION": "ПОВТОРЕНИЕ"},
    "Ukraiński": {"Polski": "Польська", "Angielski": "Англійська", "Niemiecki": "Німецька", "Holenderski": "Нідерландська", "Francuski": "Французька", "Hiszpański": "Іспанська", "Włoski": "Італійська", "Portugalski": "Португальська", "Rosyjski": "Російська", "Ukraiński": "Українська", "Białoruski": "Білоруська", "Słowacki": "Словацька", "Czeski": "Чеська", "Ustawienia": "Налаштування", "Język interfejsu aplikacji:": "Мова інтерфейсу:", "Dostawca modelu AI:": "Провайдер ШІ:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Платна опція тимчасово недоступна.", "Wybierz zadanie:": "Виберіть завдання:", "Groq (Darmowe chmurowe)": "Groq (Безкоштовна хмара)", "OpenAI (Płatne)": "OpenAI (Платний)", "Tłumaczenie": "Переклад", "Korekta tekstu": "Корекція тексту", "Generowanie audio": "Генерація аудіо", "Zapytanie": "Запит", "Notatki": "Нотатки", "Powtórki": "Картки", "Historia i Zapisane": "Історія та збережене", "Wprowadź tekst do przetłumaczenia": "Введіть текст для перекладу", "Twój tekst:": "Ваш текст:", "Język docelowy": "Цільова мова", "Tłumacz": "Перекласти", "Trwa tłumaczenie...": "Переклад...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Ви перекладач. Поверніть лише чистий переклад.", "Korekta błędów i redakcja": "Виправлення помилок", "Tekst do poprawy:": "Текст для виправлення:", "Popraw tekst": "Виправити текст", "Analiza błędów...": "Аналіз помилок...", "Popraw błędy. Zwróć poprawiony tekst.": "Виправте помилки.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Потім по пунктах поясніть внесені зміни.", "Zamiana tekstu na mowę": "Текст у мову", "Wpisz tekst:": "Введіть текст:", "Generuj wymowę": "Згенерувати вимову", "Przygotowywanie...": "Підготовка...", "Zapisz to Audio do Historii": "Зберегти аудіо", "Wygenerowane TTS": "Згенероване TTS", "Zapisano audio w historii!": "Аудіо збережено в історії!", "Czat z ekspertem językowym": "Чат з мовним експертом", "Zadaj pytanie:": "Поставте запитання:", "Zapytaj": "Запитати", "Piszę odpowiedź...": "Пишу відповідь...", "Jesteś ekspertem językowym.": "Ви мовний експерт.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Зберігайте мову і текст із семантичним пошуком", "🎤 Nagraj notatkę głosową": "🎤 Записати нотатку", "Opcjonalnie: Transkrybuj mowę": "Транскрибувати мову", "Rozpoznawanie mowy...": "Розпізнавання мови...", "Błąd transkrypcji": "Помилка транскрипції", "Treść notatki tekstowej (lub transkrypcji):": "Вміст нотатки:", "Zapisz Notatkę do bazy": "Зберегти нотатку", "Notatka jest pusta!": "Нотатка порожня!", "Zapisywanie i generowanie wektorów...": "Збереження...", "Notatka zapisana pomyślnie!": "Нотатку успішно збережено!", "Twój system nauki (Spaced Repetition)": "Ваша система навчання", "Nauka (SM-2)": "Навчання (SM-2)", "Dodaj Fiszki": "Додати картки", "Edytuj / Usuń": "Редактировать / Видалити", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Вітаємо! На сьогодні слів немає.", "Do powtórzenia": "Для повторення", "Język": "Мова", "Baza": "База", "Gramatyka": "Граматика", "Pokaż odpowiedź": "Показати відповідь", "Definicje": "Визначення", "Odmiany": "Відмінювання", "Przykłady": "Приклади", "Nie pamiętam": "Не пам'ятаю", "Z trudem": "З труднощами", "Dobrze": "Добре", "Z łatwością": "Легко", "Kreator z analizą lingwistyczną AI": "Творець з аналізом ШІ", "Wpisz słowo lub wyrażenie:": "Введіть слово:", "Język słowa:": "Мова слова:", "Pobierz dane przez AI": "Отримати дані ШІ", "AI analizuje słowo jako API...": "ШІ аналізує слово...", "Pobierz zdania z bazy Tatoeba": "Отримати речення Tatoeba", "Szukam autentycznych przykładów...": "Пошук прикладів...", "API nie znalazło przykładów.": "API не знайшло прикладів.", "Wprowadź ręcznie": "Ввести вручную", "Forma podstawowa": "Початкова форма", "Info gramatyczne": "Граматична інфо", "Wybierz definicje do zapisania:": "Виберіть визначення:", "Wybierz przykłady zdań:": "Виберіть приклади:", "Zapisz gotową fiszkę": "Зберегти картку", "Fiszka dodana pomyślnie!": "Картку додано!", "Twoje własne uwagi (opcjonalnie):": "Ваші власні нотатки:", "Twoje uwagi": "Ваші нотатки", "Język tłumaczenia (definicje):": "Мова перекладу:", "Słowo": "Слово", "💾 Zapisz zmiany": "💾 Зберегти зміни", "Audio (opcjonalnie):": "Аудіо (необов'язково):", "Generuj wymowę (TTS)": "Згенерувати вимову", "🎤 Nagraj wymowę": "🎤 Записати вимову", "Usuń przypisane audio": "Видалити аудіо", "Następna": "Наступная", "🗑 Usuń fiszkę": "🗑 Видалити картку", "Centrum Danych i Historii": "Центр даних та історії", "Pełna Historia Aktywności": "Повна історія активності", "Baza Notatek (Wyszukiwanie Semantyczne)": "База нотаток", "Brak historii aktywności.": "Немає історії активності.", "Wejście": "Вхід", "Wyjście": "Вихід", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Семантичний пошук неактивний.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Шукати в нотатках:", "Typ": "Тип", "📝 Zapisz jako notatkę": "📝 Зберегти як нотатку", "📇 Przekaż do fiszek": "📇 Відправити до карток", "🔊 Przekaż do TTS": "🔊 Відправити до TTS", "Zapisano w notatkach!": "Збережено в нотатках!", "Zalogowano pomyślnie": "Успішний вхід", "Wyloguj": "Вийти", "Zaloguj się do Lingilo": "Увійти до Lingilo", "Zaloguj przez Auth0": "Увійти через Auth0", "Błąd komunikacji z Auth0": "Помилка зв'язку з Auth0", "Błąd AI": "Помилка ШІ", "Błąd TTS": "Помилка TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Ваш помічник", "Własne definicje (każda w nowej linii - opcjonalnie):": "Власні визначення:", "np. definicja 1\ndefinicja 2": "напр. визн 1\nвизн 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Власні приклади:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "напр. Приклад 1\nПриклад 2", "TEXT": "ТЕКСТ", "AUDIO": "АУДІО", "GENERATED_AUDIO": "ЗГЕНЕРАВАНАЕ АУДІО", "TRANSLATION": "ПЕРЕКЛАД", "CORRECTION": "КОРЕКЦІЯ", "TEXT_TO_SPEECH": "ТЕКСТ У МОВУ", "INQUIRY": "ЗАПИТ", "REVISION": "ПОВТОРЕННЯ"},
    "Białoruski": {"Polski": "Польская", "Angielski": "Англійская", "Niemiecki": "Нямецкая", "Holenderski": "Галандская", "Francuski": "Французская", "Hiszpański": "Іспанская", "Włoski": "Італьянская", "Portugalski": "Партугальская", "Rosyjski": "Руская", "Ukraiński": "Украінская", "Białoruski": "Беларуская", "Słowacki": "Славацкая", "Czeski": "Чэшская", "Ustawienia": "Налады", "Język interfejsu aplikacji:": "Мова інтэрфейсу:", "Dostawca modelu AI:": "Правайдэр ШІ:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Платная опцыя часова недаступная.", "Wybierz zadanie:": "Выберыце задачу:", "Groq (Darmowe chmurowe)": "Groq (Бясплатнае воблака)", "OpenAI (Płatne)": "OpenAI (Платны)", "Tłumaczenie": "Пераклад", "Korekta tekstu": "Карэкцыя тэксту", "Generowanie audio": "Генерацыя аўдыё", "Zapytanie": "Запыт", "Notatki": "Нататкі", "Powtórki": "Карткі", "Historia i Zapisane": "Гісторыя і захаванае", "Wprowadź tekst do przetłumaczenia": "Увядзіце тэкст dla перакладу", "Twój tekst:": "Ваш тэкст:", "Język docelowy": "Мэтавая мова", "Tłumacz": "Перакласці", "Trwa tłumaczenie...": "Ідзе пераклад...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Вы перакладчык. Вярніце толькі чысты пераклад.", "Korekta błędów i redakcja": "Выпраўленне памылак", "Tekst do poprawy:": "Тэкст для выпраўлення:", "Popraw tekst": "Выправіць тэкст", "Analiza błędów...": "Аналіз памылак...", "Popraw błędy. Zwróć poprawiony tekst.": "Выпраўце памылкі.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Затым па пунктах растлумачце змены.", "Zamiana tekstu na mowę": "Тэкст у маўленне", "Wpisz tekst:": "Увядзіце тэкст:", "Generuj wymowę": "Згенераваць вымаўленне", "Przygotowywanie...": "Падрыхтоўка...", "Zapisz to Audio do Historii": "Захаваць аўдыё ў гісторыю", "Wygenerowane TTS": "Згенераванае TTS", "Zapisano audio w historii!": "Аўдыё захавана!", "Czat z ekspertem językowym": "Чат з экспертам", "Zadaj pytanie:": "Спытаць:", "Zapytaj": "Спытаць", "Piszę odpowiedź...": "Пішу адказ...", "Jesteś ekspertem językowym.": "Вы моўны эксперт.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Захоўвайце з семантычным пошукам", "🎤 Nagraj notatkę głosową": "🎤 Запісаць нататку", "Opcjonalnie: Transkrybuj mowę": "Транскрыбаваць", "Rozpoznawanie mowy...": "Распазнаванне маўлення...", "Błąd transkrypcji": "Памылка транскрыпцыі", "Treść notatki tekstowej (lub transkrypcji):": "Змест нататкі:", "Zapisz Notatkę do bazy": "Захаваць нататку", "Notatka jest pusta!": "Нататка пустая!", "Zapisywanie i generowanie wektorów...": "Захаванне...", "Notatka zapisana pomyślnie!": "Нататка паспяхова захавана!", "Twój system nauki (Spaced Repetition)": "Ваша сістэма навучання", "Nauka (SM-2)": "Навучанне (SM-2)", "Dodaj Fiszki": "Дадаць карткі", "Edytuj / Usuń": "Рэдагаваць / Выдаліць", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Віншуем! Няма слоў на сёння.", "Do powtórzenia": "Да паўтарэння", "Język": "Мова", "Baza": "База", "Gramatyka": "Граматыка", "Pokaż odpowiedź": "Паказаць адказ", "Definicje": "Азначэнні", "Odmiany": "Скланенні", "Przykłady": "Прыклады", "Nie pamiętam": "Не памятаю", "Z trudem": "З цяжкасцю", "Dobrze": "Добра", "Z łatwością": "Лёгка", "Kreator z analizą lingwistyczną AI": "Стваральнік з ІШ-аналізам", "Wpisz słowo lub wyrażenie:": "Увядзіце слова:", "Język słowa:": "Мова слова:", "Pobierz dane przez AI": "Атрымаць дадзеныя (ІШ)", "AI analizuje słowo jako API...": "ІШ аналізуе слова...", "Pobierz zdania z bazy Tatoeba": "Атрымаць сказы Tatoeba", "Szukam autentycznych przykładów...": "Пошук прыкладаў...", "API nie znalazło przykładów.": "API не знайшло прыкладаў.", "Wprowadź ręcznie": "Увесці ўручную", "Forma podstawowa": "Пачатковая форма", "Info gramatyczne": "Граматычная інфо", "Wybierz definicje do zapisania:": "Выберыце азначэнні:", "Wybierz przykłady zdań:": "Выберыце прыклады:", "Zapisz gotową fiszkę": "Захаваць картку", "Fiszka dodana pomyślnie!": "Картка дададзена!", "Twoje własne uwagi (opcjonalnie):": "Вашы ўласныя нататкі:", "Twoje uwagi": "Вашы нататкі", "Język tłumaczenia (definicje):": "Мова перакладу:", "Słowo": "Слова", "💾 Zapisz zmiany": "💾 Захаваць змены", "Audio (opcjonalnie):": "Аўдыё (неабавязкова):", "Generuj wymowę (TTS)": "Згенераваць вымаўленне", "🎤 Nagraj wymowę": "🎤 Запісаць вымаўленне", "Usuń przypisane audio": "Выдаліць аўдыё", "Następna": "Наступная", "🗑 Usuń fiszkę": "🗑 Выдаліць картку", "Centrum Danych i Historii": "Цэнтр дадзеных", "Pełna Historia Aktywności": "Поўная гісторыя", "Baza Notatek (Wyszukiwanie Semantyczne)": "База нататак", "Brak historii aktywności.": "Няма гісторыі.", "Wejście": "Уваход", "Wyjście": "Выхад", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Семантычны пошук неактыўны.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Шукаць у нататках:", "Typ": "Тып", "📝 Zapisz jako notatkę": "📝 Захаваць як нататку", "📇 Przekaż do fiszek": "📇 Адправіць у карткі", "🔊 Przekaż do TTS": "🔊 Адправіць у TTS", "Zapisano w notatkach!": "Захавана ў нататках!", "Zalogowano pomyślnie": "Паспяховы ўваход", "Wyloguj": "Выйсці", "Zaloguj się do Lingilo": "Увайсці ў Lingilo", "Zaloguj przez Auth0": "Увайсці праз Auth0", "Błąd komunikacji z Auth0": "Памылка сувязі з Auth0", "Błąd AI": "Памылка ШІ", "Błąd TTS": "Памылка TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Ваш памочнік", "Własne definicje (każda w nowej linii - opcjonalnie):": "Уласныя азначэнні:", "np. definicja 1\ndefinicja 2": "напр. азначэнне 1\nазначэнне 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Уласныя прыклады:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "напр. Прыклад 1\nПрыклад 2", "TEXT": "ТЭКСТ", "AUDIO": "АЎДЫЁ", "GENERATED_AUDIO": "ЗГЕНЕРАВАНАЕ АЎДЫЁ", "TRANSLATION": "ПЕРАКЛАД", "CORRECTION": "КАРЭКЦЫЯ", "TEXT_TO_SPEECH": "ТЭКСТ У МАЎЛЕННЕ", "INQUIRY": "ЗАПЫТ", "REVISION": "ПАЎТАРЭННЕ"},
    "Słowacki": {"Polski": "Poľština", "Angielski": "Angličtina", "Niemiecki": "Nemčina", "Holenderski": "Holandčina", "Francuski": "Francúzština", "Hiszpański": "Španielčina", "Włoski": "Taliančina", "Portugalski": "Portugalčina", "Rosyjski": "Ruština", "Ukraiński": "Ukrajinčina", "Białoruski": "Bieloruština", "Słowacki": "Slovenčina", "Czeski": "Čeština", "Ustawienia": "Nastavenia", "Język interfejsu aplikacji:": "Jazyk rozhrania:", "Dostawca modelu AI:": "Poskytovateľ AI:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Platená verzia nie je dostupná.", "Wybierz zadanie:": "Vyberte úlohu:", "Groq (Darmowe chmurowe)": "Groq (Bezplatný Cloud)", "OpenAI (Płatne)": "OpenAI (Platené)", "Tłumaczenie": "Preklad", "Korekta tekstu": "Oprava textu", "Generowanie audio": "Generovanie zvuku", "Zapytanie": "Dopyt", "Notatki": "Poznámky", "Powtórki": "Kartičky", "Historia i Zapisane": "História a Uložené", "Wprowadź tekst do przetłumaczenia": "Zadajte text na preklad", "Twój tekst:": "Váš text:", "Język docelowy": "Cieľový jazyk", "Tłumacz": "Preložiť", "Trwa tłumaczenie...": "Prekladám...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Ste prekladateľ. Vráťte iba preklad.", "Korekta błędów i redakcja": "Oprava chýb", "Tekst do poprawy:": "Text na opravu:", "Popraw tekst": "Opraviť text", "Analiza błędów...": "Analýza chýb...", "Popraw błędy. Zwróć poprawiony tekst.": "Opravte chyby. Vráťte opravený text.", "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Potom v bodoch vysvetlite zmeny a zdôvodnite.", "Zamiana tekstu na mowę": "Text na reč", "Wpisz tekst:": "Zadajte text:", "Generuj wymowę": "Generovať výslovnosť", "Przygotowywanie...": "Pripravujem...", "Zapisz to Audio do Historii": "Uložiť zvuk do histórie", "Wygenerowane TTS": "Generované TTS", "Zapisano audio w historii!": "Zvuk uložený!", "Czat z ekspertem językowym": "Chat s expertom", "Zadaj pytanie:": "Položte otázku:", "Zapytaj": "Spýtať sa", "Piszę odpowiedź...": "Píšem odpoveď...", "Jesteś ekspertem językowym.": "Ste jazykový expert.", "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Uložte reč s sémantickým vyhľadávaním", "🎤 Nagraj notatkę głosową": "🎤 Nahrať hlasovú poznámku", "Opcjonalnie: Transkrybuj mowę": "Voliteľné: Prepísať reč", "Rozpoznawanie mowy...": "Rozpoznávanie reči...", "Błąd transkrypcji": "Chyba prepisu", "Treść notatki tekstowej (lub transkrypcji):": "Obsah poznámky:", "Zapisz Notatkę do bazy": "Uložiť poznámku", "Notatka jest pusta!": "Poznámka je prázdna!", "Zapisywanie i generowanie wektorów...": "Ukladám...", "Notatka zapisana pomyślnie!": "Poznámka uložená!", "Twój system nauki (Spaced Repetition)": "Váš výučbový systém", "Nauka (SM-2)": "Štúdium (SM-2)", "Dodaj Fiszki": "Pridať kartičky", "Edytuj / Usuń": "Upraviť / Zmazať", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Gratulujeme! Dnes žiadne slovíčka.", "Do powtórzenia": "Na opakovanie", "Język": "Jazyk", "Baza": "Základ", "Gramatyka": "Gramatika", "Pokaż odpowiedź": "Ukázať odpoveď", "Definicje": "Definície", "Odmiany": "Skloňovanie", "Przykłady": "Príklady", "Nie pamiętam": "Nepamätám", "Z trudem": "Ťažko", "Dobrze": "Dobre", "Z łatwością": "Ľahko", "Kreator z analizą lingwistyczną AI": "Tvorca s AI analýzou", "Wpisz słowo lub wyrażenie:": "Zadajte slovo:", "Język słowa:": "Jazyk slova:", "Pobierz dane przez AI": "Získať dáta z AI", "AI analizuje słowo jako API...": "AI analyzuje slovo...", "Pobierz zdania z bazy Tatoeba": "Získať vety z Tatoeba", "Szukam autentycznych przykładów...": "Hľadám príklady...", "API nie znalazło przykładów.": "API nenašlo príklady.", "Wprowadź ręcznie": "Zadať ručne", "Forma podstawowa": "Základná forma", "Info gramatyczne": "Gramatické info", "Wybierz definicje do zapisania:": "Vyberte definície:", "Wybierz przykłady zdań:": "Vyberte vety:", "Zapisz gotową fiszkę": "Uložiť kartičku", "Fiszka dodana pomyślnie!": "Kartička pridaná!", "Twoje własne uwagi (opcjonalnie):": "Vaše vlastné poznámky:", "Twoje uwagi": "Vaše poznámky", "Język tłumaczenia (definicje):": "Jazyk prekladu (definície):", "Słowo": "Slovo", "💾 Zapisz zmiany": "💾 Uložiť zmeny", "Audio (opcjonalnie):": "Zvuk (voliteľné):", "Generuj wymowę (TTS)": "Generovať výslovnosť", "🎤 Nagraj wymowę": "🎤 Nahrať výslovnost", "Usuń przypisane audio": "Odstrániť zvuk", "Następna": "Ďalšia", "🗑 Usuń fiszkę": "🗑 Zmazať kartičku", "Centrum Danych i Historii": "Dátové centrum", "Pełna Historia Aktywności": "Kompletná história", "Baza Notatek (Wyszukiwanie Semantyczne)": "Poznámky (Sémantické hľadanie)", "Brak historii aktywności.": "Žiadna história.", "Wejście": "Vstup", "Wyjście": "Výstup", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Sémantické hľadanie neaktívne.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Hľadať v poznámkách:", "Typ": "Typ", "📝 Zapisz jako notatkę": "📝 Uložiť ako poznámku", "📇 Przekaż do fiszek": "📇 Odoslať do kartiček", "🔊 Przekaż do TTS": "🔊 Odoslať do TTS", "Zapisano w notatkach!": "Uložené v poznámkách!", "Zalogowano pomyślnie": "Úspešne prihlásené", "Wyloguj": "Odhlásiť sa", "Zaloguj się do Lingilo": "Prihláste sa do Lingilo", "Zaloguj przez Auth0": "Prihlásiť sa cez Auth0", "Błąd komunikacji z Auth0": "Chyba komunikácie s Auth0", "Błąd AI": "Chyba AI", "Błąd TTS": "Chyba TTS", "Lingilo - Twój pomocnik językowy": "Lingilo - Váš asistent", "Własne definicje (każda w nowej linii - opcjonalnie):": "Vlastní definície:", "np. definicja 1\ndefinicja 2": "napr. def 1\ndef 2", "Własne przykłady (każdy w nowej linii - opcjonalnie):": "Vlastné príklady:", "np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład.": "napr. Príklad 1\nPríklad 2", "TEXT": "TEXT", "AUDIO": "AUDIO", "GENERATED_AUDIO": "GENEROVANÉ AUDIO", "TRANSLATION": "PREKLAD", "CORRECTION": "OPRAVA", "TEXT_TO_SPEECH": "TEXT NA REČ", "INQUIRY": "DOPYT", "REVISION": "OPAKOVANIE"}
}

async def load_user_preferences(user_id):
    async_session = sessionmaker(bind=get_engine(), class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        res = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = res.scalars().first()
        if not prefs:
            current_lang = st.session_state.get('ui_language', 'Polski')
            prefs = UserPreferences(user_id=user_id, ui_language=current_lang)
            session.add(prefs)
            await session.commit()
        return prefs

async def update_user_preference(user_id, key, value):
    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
        res = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = res.scalars().first()
        if prefs:
            setattr(prefs, key, value)
            await session.commit()

async def save_note_to_db(text):
    emb_json = None
    if SEMANTIC_SEARCH_AVAILABLE and text:
        emb = embedder.encode(text).tolist()
        emb_json = json.dumps(emb)
    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
        session.add(Note(user_id=st.session_state.current_user_id, text=text, note_type=NoteType.TEXT, embedding=emb_json))
        await session.commit()

async def save_current_drafts(current_user):
    if not current_user or not st.session_state.get('prefs_loaded'): return
    
    drafts_to_save = st.session_state.p_state.copy()
    current_drafts_json = json.dumps(drafts_to_save)
    if current_drafts_json != st.session_state.get('last_saved_drafts'):
        await update_user_preference(current_user, "session_drafts", current_drafts_json)
        st.session_state.last_saved_drafts = current_drafts_json

def calculate_sm2(quality: int, repetition: int, easiness_factor: float, interval: int):
    if quality >= 3:
        if repetition == 0: interval = 1
        elif repetition == 1: interval = 6
        else: interval = int(round(interval * easiness_factor))
        repetition += 1
    else:
        repetition = 0
        interval = 1
    easiness_factor = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if easiness_factor < 1.3: easiness_factor = 1.3
    return repetition, easiness_factor, interval

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

def get_example_sentences_api(word, language_name, target_language_name):
    try:
        from_lang_code = TATOEBA_LANG_MAP.get(language_name, "eng")
        dest_lang_code = TATOEBA_LANG_MAP.get(target_language_name, "pol")
        
        url = f"https://dev.tatoeba.org/pl/api_v0/search?from={from_lang_code}&query={word}&trans_filter=1&trans_to={dest_lang_code}"
        
        headers = {"User-Agent": "Lingilo-App/1.0"}
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            examples = []
            
            for result in data.get("results", []):
                src = result.get("text", "").strip()
                tgt = ""
                
                translations = result.get("translations", [])
                if translations and isinstance(translations, list):
                    for trans_group in translations:
                        if isinstance(trans_group, list) and len(trans_group) > 0:
                            tgt = trans_group[0].get("text", "").strip()
                            break
                
                if src and tgt:
                    examples.append(f"{src} ➔ {tgt}")
                    if len(examples) == 5: break
                        
            return examples
        else: return []
    except Exception as e:
        print(f"Błąd API Tatoeba: {e}")
        return []

def get_async_openai_client():
    api_key = get_secret("GROQ_API_KEY", "BRAK_KLUCZA")
    return openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

async def transcribe_audio(audio_bytes):
    client = get_async_openai_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    try:
        res = await client.audio.transcriptions.create(file=audio_file, model="whisper-large-v3")
        return res.text
    except Exception as e:
        return f"{_('Błąd transkrypcji')}: {e}"

# ==========================================
# ASYNCHRONICZNY RDZEŃ APLIKACJI
# ==========================================
async def app_core(current_user, cookie_manager):

    # POPRAWIONY CALLBACK: Bezpieczny dla nowej struktury main()
    def process_logout():
        # 1. Zapisywanie stanu / kopii roboczych przed usunięciem sesji
        drafts_to_save = st.session_state.p_state.copy()
        current_drafts_json = json.dumps(drafts_to_save)
        if current_drafts_json != st.session_state.get('last_saved_drafts'):
            try:
                # Pobieramy aktualną pętlę stworzoną przez main() i wykonujemy zapis
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                
                # Uruchamiamy bezpośrednio w pętli bez zamrażania wątków chmurowych
                loop.run_until_complete(
                    update_user_preference(current_user, "session_drafts", current_drafts_json)
                )
                st.session_state.last_saved_drafts = current_drafts_json
            except Exception:
                # Fallback "odpal i zapomnij" przez create_task, jeśli pętla główna jest w stanie zapisu
                try:
                    loop.create_task(update_user_preference(current_user, "session_drafts", current_drafts_json))
                except Exception:
                    pass

        # 2. Czyszczenie ciasteczka w przeglądarce
        try:
            cookie_manager.set("auth0_token", "", key="clear_cookie_on_manual_logout")
        except Exception:
            pass

        # 3. Czyszczenie stanu sesji Pythona
        if "token" in st.session_state:
            del st.session_state.token
        if "user_info" in st.session_state:
            del st.session_state.user_info
            
        st.session_state.prefs_loaded = False
        st.query_params.clear()
        
        # BLOKADA "ZOMBIE COOKIE"
        st.session_state.logged_out_flag = True

    # --- INICJALIZACJA BAZY I PREFERENCJI (BEZ ZMIAN) ---
    if "db_initialized" not in st.session_state:
        await create_tables()
        st.session_state.db_initialized = True

    if not st.session_state.prefs_loaded:
        prefs = await load_user_preferences(current_user)
        st.session_state.ui_language = prefs.ui_language
        st.session_state.translation_target_language = prefs.translation_target_language
        st.session_state.ai_provider = prefs.ai_provider
        
        if prefs.session_drafts:
            try:
                drafts = json.loads(prefs.session_drafts)
                st.session_state.p_state.update(drafts)
                st.session_state.last_saved_drafts = prefs.session_drafts
            except Exception: pass
                
        st.session_state.prefs_loaded = True

    # --- PASEK BOCZNY ---
    with st.sidebar:
        st.success("✅ " + _("Zalogowano pomyślnie"))
        
        # Przycisk odpala cały proces w callbacku, a po kliknięciu robi czysty rerun
        if st.button(_("Wyloguj"), on_click=process_logout):
            st.rerun()

        st.header("⚙️ " + _("Ustawienia"))
        
        ui_langs_display = [NATIVE_LANGUAGE_NAMES[l] for l in SUPPORTED_LANGUAGES]
        sel_ui_trans = st.selectbox(
            _("Język interfejsu aplikacji:"), 
            ui_langs_display, 
            index=SUPPORTED_LANGUAGES.index(st.session_state.ui_language),
            key="sidebar_language_picker"
        )
        new_ui_lang = SUPPORTED_LANGUAGES[ui_langs_display.index(sel_ui_trans)]
        
        if new_ui_lang != st.session_state.ui_language:
            st.session_state.ui_language = new_ui_lang
            await update_user_preference(current_user, "ui_language", new_ui_lang)
            st.rerun()
            
        provider_options = ["Groq (Darmowe chmurowe)", "OpenAI (Płatne)"]
        prov_translated = [_(p) for p in provider_options]
        
        try: prov_idx = provider_options.index(st.session_state.ai_provider)
        except ValueError: prov_idx = 0
            
        sel_prov_trans = st.selectbox(
            _("Dostawca modelu AI:"), 
            prov_translated, 
            index=prov_idx
        )
        new_prov = provider_options[prov_translated.index(sel_prov_trans)]
        
        # SPRAWDZENIE KLUCZA I WYŚWIETLENIE KOMUNIKATU W MENU
        if new_prov == "OpenAI (Płatne)":
            openai_key = get_secret("OPENAI_API_KEY")
            if not openai_key or openai_key == "BRAK_KLUCZA":
                st.warning(_("Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq."))
                new_prov = "Groq (Darmowe chmurowe)"

        if new_prov != st.session_state.ai_provider:
            st.session_state.ai_provider = new_prov
            await update_user_preference(current_user, "ai_provider", new_prov)
            st.rerun()
        
        task_options = [
            "Tłumaczenie", "Korekta tekstu", "Generowanie audio", 
            "Zapytanie", "Notatki", "Powtórki", "Historia i Zapisane"
        ]
        task_translated = [_(t) for t in task_options]
        
        try: current_idx = task_options.index(st.session_state.active_task)
        except ValueError: current_idx = 0
            
        sel_task_trans = st.radio(
            _("Wybierz zadanie:"), 
            task_translated,
            index=current_idx
        )
        task = task_options[task_translated.index(sel_task_trans)]
        
        if task != st.session_state.active_task:
            drafts_to_save = st.session_state.p_state.copy()
            current_drafts_json = json.dumps(drafts_to_save)
            if current_drafts_json != st.session_state.get('last_saved_drafts'):
                await update_user_preference(current_user, "session_drafts", current_drafts_json)
                st.session_state.last_saved_drafts = current_drafts_json
            st.session_state.active_task = task
            st.rerun()

    current_task = st.session_state.active_task

    if current_task == "Tłumaczenie":
        st.header(_("Wprowadź tekst do przetłumaczenia"))
        
        p_text_area(_("Twój tekst:"), key="trans_text")
        
        lang_trans = [_(l) for l in SUPPORTED_LANGUAGES]
        sel_lang_trans = st.selectbox(_("Język docelowy"), lang_trans, index=1)
        lang = SUPPORTED_LANGUAGES[lang_trans.index(sel_lang_trans)]

        if st.button(_("Tłumacz")) and st.session_state.p_state['trans_text']:
            with st.spinner(_("Trwa tłumaczenie...")):
                st.session_state.translated_text = await monitor_llm(_("Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie."), f"Na {lang}:\n{st.session_state.p_state['trans_text']}", TaskType.TRANSLATION)
        
        if st.session_state.translated_text:
            st.success(st.session_state.translated_text)
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            
            if c1.button(_("📝 Zapisz jako notatkę")):
                note_content = f"Tłumaczenie ({lang}):\n{st.session_state.p_state['trans_text']}\n➔\n{st.session_state.translated_text}"
                await save_note_to_db(note_content)
                st.toast(_("Zapisano w notatkach!"), icon="✅")
                
            if c2.button(_("📇 Przekaż do fiszek")):
                st.session_state.p_state['flashcard_input_word'] = st.session_state.p_state['trans_text']
                st.session_state.active_task = "Powtórki"
                st.session_state.flashcard_sub_tab = "add"
                st.rerun()
                
            if c3.button(_("🔊 Przekaż do TTS")):
                st.session_state.p_state['tts_text'] = st.session_state.translated_text
                st.session_state.active_task = "Generowanie audio"
                st.rerun()

    elif current_task == "Korekta tekstu":
        st.header(_("Korekta błędów i redakcja"))
        p_text_area(_("Tekst do poprawy:"), key="corr_text")
        
        if st.button(_("Popraw tekst")) and st.session_state.p_state['corr_text']:
            with st.spinner(_("Analiza błędów...")):
                sys_prompt = _("Popraw błędy. Zwróć poprawiony tekst.") + " " + _("Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.")
                st.session_state.corrected_text = await monitor_llm(sys_prompt, st.session_state.p_state['corr_text'], TaskType.CORRECTION)
        
        if st.session_state.corrected_text:
            st.info(st.session_state.corrected_text)
            st.markdown("---")
            c1, c2 = st.columns(2)
            if c1.button(_("📝 Zapisz jako notatkę")):
                note_content = f"Korekta:\n{st.session_state.p_state['corr_text']}\n\nAnaliza:\n{st.session_state.corrected_text}"
                await save_note_to_db(note_content)
                st.toast(_("Zapisano w notatkach!"), icon="✅")
                
            if c2.button(_("🔊 Przekaż do TTS")):
                st.session_state.p_state['tts_text'] = st.session_state.corrected_text
                st.session_state.active_task = "Generowanie audio"
                st.rerun()

    elif current_task == "Generowanie audio":
        st.header(_("Zamiana tekstu na mowę"))
        p_text_area(_("Wpisz tekst:"), key="tts_text")
        
        col1, col2 = st.columns(2)
        if col1.button(_("Generuj wymowę")) and st.session_state.p_state['tts_text']:
            with st.spinner(_("Przygotowywanie...")):
                try: lang_code = detect(st.session_state.p_state['tts_text'])
                except: lang_code = "en"
                
                try: tts = gTTS(text=st.session_state.p_state['tts_text'], lang=lang_code)
                except ValueError: tts = gTTS(text=st.session_state.p_state['tts_text'], lang="en")
                    
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                st.session_state.audio_generation_audio_file = buf.getvalue()
                
        if st.session_state.audio_generation_audio_file:
            st.audio(st.session_state.audio_generation_audio_file, format="audio/mp3")
            if col2.button(_("Zapisz to Audio do Historii")):
                with st.spinner(_("Zapisywanie i generowanie wektorów...")):
                    emb_json = None
                    if SEMANTIC_SEARCH_AVAILABLE and st.session_state.p_state['tts_text']:
                        emb = embedder.encode(st.session_state.p_state['tts_text']).tolist()
                        emb_json = json.dumps(emb)

                    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                        session.add(Note(
                            user_id=current_user,
                            text=f"{_('Wygenerowane TTS')}: {st.session_state.p_state['tts_text'][:50]}...", 
                            note_type=NoteType.GENERATED_AUDIO, 
                            audio_data=st.session_state.audio_generation_audio_file,
                            embedding=emb_json
                        ))
                        await session.commit()
                st.toast(_("Zapisano audio w historii!"), icon="✅")

    elif current_task == "Zapytanie":
        st.header(_("Czat z ekspertem językowym"))
        p_text_area(_("Zadaj pytanie:"), key="inq_text")
        
        if st.button(_("Zapytaj")) and st.session_state.p_state['inq_text']:
            with st.spinner(_("Piszę odpowiedź...")):
                st.session_state.inquiry_response = await monitor_llm(_("Jesteś ekspertem językowym."), st.session_state.p_state['inq_text'], TaskType.INQUIRY)
        
        if st.session_state.inquiry_response:
            st.info(st.session_state.inquiry_response)
            if st.button(_("📝 Zapisz jako notatkę")):
                note_content = f"Pytanie: {st.session_state.p_state['inq_text']}\n\nOdpowiedź:\n{st.session_state.inquiry_response}"
                await save_note_to_db(note_content)
                st.toast(_("Zapisano w notatkach!"), icon="✅")

    elif current_task == "Notatki":
        st.header(_("Zapisuj mowę i tekst z wyszukiwaniem semantycznym"))
        
        if "note_audio_key" not in st.session_state:
            st.session_state.note_audio_key = f"note_audio_{uuid.uuid4()}"
            
        note_audio = st.audio_input(_("🎤 Nagraj notatkę głosową"), key=st.session_state.note_audio_key)
        
        current_bytes = note_audio.getvalue() if note_audio else None
        
        if current_bytes and current_bytes != st.session_state.get("last_processed_audio_bytes"):
            st.session_state.last_processed_audio_bytes = current_bytes
            st.session_state.note_audio_bytes = current_bytes
            
            st.session_state.p_state['note_audio_text'] = ""
            st.session_state.pop("wid_note_audio_text", None)
            st.rerun()

        if st.session_state.get("note_audio_bytes"):
            st.success("Plik audio gotowy do przetworzenia:")
            st.audio(st.session_state.note_audio_bytes)
            
            c1, c2 = st.columns(2)
            if c1.button("🗑 Usuń nagranie", key="del_note_audio"):
                st.session_state.note_audio_bytes = None
                st.session_state.last_processed_audio_bytes = None
                st.session_state.p_state['note_audio_text'] = ""
                st.session_state.pop("wid_note_audio_text", None)
                st.session_state.note_audio_key = f"note_audio_{uuid.uuid4()}"
                st.rerun()
                
            if c2.button(_("Opcjonalnie: Transkrybuj mowę"), key="transcribe_note_audio"):
                with st.spinner(_("Rozpoznawanie mowy...")):
                    transcription = await transcribe_audio(st.session_state.note_audio_bytes)
                    
                    # MAGIA: Przycisk jest POWYŻEJ pola, więc bezpiecznie "wstrzykujemy" tekst i pomijamy st.rerun(). 
                    # Kod swobodnie popłynie w dół i narysuje pełne okienko!
                    st.session_state.p_state['note_audio_text'] = transcription
                    st.session_state["wid_note_audio_text"] = transcription
                    
        p_text_area(_("Treść notatki tekstowej (lub transkrypcji):"), key="note_audio_text")
        
        if st.button(_("Zapisz Notatkę do bazy"), type="primary"):
            if not st.session_state.p_state['note_audio_text'] and not st.session_state.get("note_audio_bytes"):
                st.warning(_("Notatka jest pusta!"))
            else:
                with st.spinner(_("Zapisywanie i generowanie wektorów...")):
                    emb_json = None
                    if SEMANTIC_SEARCH_AVAILABLE and st.session_state.p_state['note_audio_text']:
                        emb = embedder.encode(st.session_state.p_state['note_audio_text']).tolist()
                        emb_json = json.dumps(emb)
                        
                    n_type = NoteType.AUDIO if st.session_state.get("note_audio_bytes") else NoteType.TEXT
                    
                    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                        session.add(Note(
                            user_id=current_user, 
                            text=st.session_state.p_state['note_audio_text'], 
                            note_type=n_type, 
                            audio_data=st.session_state.note_audio_bytes, 
                            embedding=emb_json
                        ))
                        await session.commit()
                st.toast(_("Notatka zapisana pomyślnie!"), icon="✅")
                
                # MAGIA 2: Przycisk Zapisz jest PONIŻEJ pola, więc tutaj nie wolno użyć "=".
                # Używamy bezpiecznego usunięcia z pamięci i resetujemy stronę.
                st.session_state.note_audio_bytes = None
                st.session_state.last_processed_audio_bytes = None
                st.session_state.p_state['note_audio_text'] = ""
                st.session_state.pop("wid_note_audio_text", None)
                st.session_state.note_audio_key = f"note_audio_{uuid.uuid4()}"
                st.rerun()

    elif current_task == "Powtórki":
            st.header(_("Twój system nauki (Spaced Repetition)"))
            
            f_options = ["study", "add", "edit"]
            f_labels = [_("Nauka (SM-2)"), _("Dodaj Fiszki"), _("Edytuj / Usuń")]
            
            current_f_idx = f_options.index(st.session_state.flashcard_sub_tab)
            
            sel_f_label = st.segmented_control(
                "Nawigacja fiszek", 
                options=f_labels, 
                default=f_labels[current_f_idx],
                selection_mode="single",
                label_visibility="collapsed"
            )
            
            if sel_f_label is None:
                st.rerun()
                
            new_f_tab = f_options[f_labels.index(sel_f_label)]
            if new_f_tab != st.session_state.flashcard_sub_tab:
                st.session_state.flashcard_sub_tab = new_f_tab
                st.rerun()
                
            st.markdown("---")
            
            if st.session_state.flashcard_sub_tab == "study":
                async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                    now = datetime.now()
                    result = await session.execute(
                        select(WordRevision).where(WordRevision.user_id == current_user)
                        .where(WordRevision.next_review_date <= now)
                        .order_by(WordRevision.next_review_date.asc())
                    )
                    due_flashcards = result.scalars().all()
                    
                    if not due_flashcards:
                        st.success(_("🎉 Gratulacje! Brak słówek na dziś."))
                    else:
                        st.write(f"**{_('Do powtórzenia')}: {len(due_flashcards)}**")
                        current_card = due_flashcards[0]
                        
                        with st.container(border=True):
                            st.title(f"🤔 {current_card.word.upper()}")
                            st.caption(f"{_('Język')}: {_(current_card.language)} | {_('Baza')}: {current_card.base_form or '-'} | {_('Gramatyka')}: {current_card.grammar_info or '-'}")
                            
                            if current_card.audio_data: st.audio(current_card.audio_data, format="audio/mp3")

                            if not st.session_state.show_flashcard_answer:
                                if st.button(_("Pokaż odpowiedź"), use_container_width=True):
                                    st.session_state.show_flashcard_answer = True
                                    st.rerun()
                            
                            if st.session_state.show_flashcard_answer:
                                st.markdown("---")
                                st.markdown(f"**{_('Definicje')}:** {current_card.definitions}")
                                st.markdown(f"**{_('Odmiany')}:** {current_card.inflected_forms}")
                                st.markdown(f"**{_('Przykłady')}:**")
                                if current_card.example_sentences:
                                    for s in str(current_card.example_sentences).split(" | "): st.write(f"- {s}")
                                    
                                if current_card.custom_notes:
                                    st.markdown(f"**{_('Twoje uwagi')}:** {current_card.custom_notes}")
                                        
                                st.markdown("---")
                                col1, col2, col3, col4 = st.columns(4)
                                grade = None
                                if col1.button(_("Nie pamiętam"), type="primary"): grade = 0
                                if col2.button(_("Z trudem")): grade = 3
                                if col3.button(_("Dobrze")): grade = 4
                                if col4.button(_("Z łatwością")): grade = 5
                                
                                if grade is not None:
                                    new_rep, new_ef, new_int = calculate_sm2(grade, current_card.repetition, current_card.easiness_factor, current_card.interval)
                                    current_card.repetition, current_card.easiness_factor, current_card.interval = new_rep, new_ef, new_int
                                    current_card.next_review_date = datetime.now() + timedelta(days=new_int)
                                    await session.commit()
                                    st.session_state.show_flashcard_answer = False
                                    st.rerun()

            elif st.session_state.flashcard_sub_tab == "add":
                st.subheader(_("Kreator z analizą lingwistyczną AI"))
                
                p_text_input(_("Wpisz słowo lub wyrażenie:"), key="flashcard_input_word")
                card_word = st.session_state.p_state['flashcard_input_word']
                
                c_lang1, c_lang2 = st.columns(2)
                with c_lang1:
                    lang_trans = [_(l) for l in SUPPORTED_LANGUAGES]
                    sel_card_lang_trans = st.selectbox(_("Język słowa:"), lang_trans)
                    card_lang = SUPPORTED_LANGUAGES[lang_trans.index(sel_card_lang_trans)]
                with c_lang2:
                    sel_target_lang_trans = st.selectbox(
                        _("Język tłumaczenia (definicje):"), 
                        lang_trans, 
                        index=SUPPORTED_LANGUAGES.index(st.session_state.translation_target_language)
                    )
                    target_lang = SUPPORTED_LANGUAGES[lang_trans.index(sel_target_lang_trans)]
                    
                    if target_lang != st.session_state.translation_target_language:
                        st.session_state.translation_target_language = target_lang
                        await update_user_preference(current_user, "translation_target_language", target_lang)
                
                c1, c2, c3 = st.columns(3)
                sync_keys = ['fc_base_form', 'fc_grammar_info', 'fc_forms', 'fc_custom_defs', 'fc_custom_sents', 'fc_custom_notes', 'fc_selected_defs', 'fc_selected_sents']

                if c1.button(_("Pobierz dane przez AI")) and card_word:
                    with st.spinner(_("AI analizuje słowo jako API...")):
                        draft = await generate_word_data(card_word, card_lang, target_lang)
                        if draft:
                            for k in sync_keys:
                                if f"wid_{k}" in st.session_state:
                                    st.session_state.p_state[k] = st.session_state[f"wid_{k}"]

                            st.session_state.current_flashcard_draft = True
                            st.session_state.p_state['fc_base_form'] = draft.get('base_form', '')
                            st.session_state.p_state['fc_grammar_info'] = draft.get('grammar_info', '')
                            forms = draft.get('forms', [])
                            st.session_state.p_state['fc_forms'] = ", ".join(forms) if isinstance(forms, list) else forms
                            
                            st.session_state.fc_def_options = draft.get('definitions', [])
                            st.session_state.p_state['fc_selected_defs'] = draft.get('definitions', [])
                            
                            ai_sents = draft.get('sentences', [])
                            curr_opts = st.session_state.get('fc_sent_options', [])
                            curr_sel = st.session_state.p_state.get('fc_selected_sents', [])
                            
                            new_sents = [s for s in ai_sents if s not in curr_opts]
                            st.session_state.fc_sent_options = curr_opts + new_sents
                            st.session_state.p_state['fc_selected_sents'] = curr_sel + new_sents
                            
                            for k in sync_keys:
                                st.session_state.pop(f"wid_{k}", None)
                            st.rerun()

                if c2.button(_("Pobierz zdania z bazy Tatoeba")) and card_word:
                    with st.spinner(_("Szukam autentycznych przykładów...")):
                        api_sents = get_example_sentences_api(card_word, card_lang, target_lang)
                        if api_sents:
                            for k in sync_keys:
                                if f"wid_{k}" in st.session_state:
                                    st.session_state.p_state[k] = st.session_state[f"wid_{k}"]

                            st.session_state.current_flashcard_draft = True
                            curr_opts = st.session_state.get('fc_sent_options', [])
                            curr_sel = st.session_state.p_state.get('fc_selected_sents', [])
                            
                            new_sents = [s for s in api_sents if s not in curr_opts]
                            if new_sents:
                                st.session_state.fc_sent_options = curr_opts + new_sents
                                st.session_state.p_state['fc_selected_sents'] = curr_sel + new_sents
                            
                            for k in sync_keys:
                                st.session_state.pop(f"wid_{k}", None)
                            st.rerun()
                        else: 
                            st.warning(_("API nie znalazło przykładów."))

                if c3.button(_("Wprowadź ręcznie")) and card_word:
                    st.session_state.current_flashcard_draft = True
                    
                    st.session_state.p_state['fc_base_form'] = card_word
                    st.session_state.p_state['fc_grammar_info'] = ""
                    st.session_state.p_state['fc_forms'] = ""
                    st.session_state.fc_def_options = []
                    st.session_state.p_state['fc_selected_defs'] = []
                    st.session_state.fc_sent_options = []
                    st.session_state.p_state['fc_selected_sents'] = []
                    
                    for k in sync_keys:
                        st.session_state.pop(f"wid_{k}", None)
                    st.rerun()
                
                if st.session_state.current_flashcard_draft:
                    st.markdown("---")
                    
                    p_text_input(_("Forma podstawowa"), key="fc_base_form")
                    p_text_input(_("Info gramatyczne"), key="fc_grammar_info")
                    p_text_input(_("Odmiany"), key="fc_forms")
                    
                    p_multiselect(_("Wybierz definicje do zapisania:"), key="fc_selected_defs", options=st.session_state.fc_def_options)
                    p_text_area(_("Własne definicje (każda w nowej linii - opcjonalnie):"), placeholder=_("np. definicja 1\ndefinicja 2"), key="fc_custom_defs")
                    
                    p_multiselect(_("Wybierz przykłady zdań:"), key="fc_selected_sents", options=st.session_state.fc_sent_options)
                    p_text_area(_("Własne przykłady (każdy w nowej linii - opcjonalnie):"), placeholder=_("np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład."), key="fc_custom_sents")
                    
                    p_text_area(_("Twoje własne uwagi (opcjonalnie):"), key="fc_custom_notes")

                    st.write(f"**{_('Audio (opcjonalnie):')}**")
                    a_col1, a_col2 = st.columns(2)
                    with a_col1:
                        if st.button(_("Generuj wymowę (TTS)")):
                            with st.spinner(_("Generowanie...")):
                                tts_lang_code = TTS_LANG_MAP.get(card_lang, "en")
                                try:
                                    tts = gTTS(text=card_word, lang=tts_lang_code)
                                    buf = io.BytesIO()
                                    tts.write_to_fp(buf)
                                    st.session_state.flashcard_audio_bytes = buf.getvalue()
                                except Exception as e:
                                    st.error(f"{_('Błąd TTS')}: {e}")
                    
                    with a_col2:
                        # --- ROZWIĄZANIE OSTATECZNE DLA FISZEK: Dynamiczny klucz widgetu ---
                        if "fc_audio_key" not in st.session_state:
                            st.session_state.fc_audio_key = f"fc_audio_{uuid.uuid4()}"
                            
                        mic_audio = st.audio_input(_("🎤 Nagraj wymowę"), key=st.session_state.fc_audio_key)
                        
                        fc_current_bytes = mic_audio.getvalue() if mic_audio else None
                        if fc_current_bytes and fc_current_bytes != st.session_state.get("last_fc_audio_bytes"):
                            st.session_state.last_fc_audio_bytes = fc_current_bytes
                            st.session_state.flashcard_audio_bytes = fc_current_bytes
                            st.rerun()
                    
                    if st.session_state.flashcard_audio_bytes:
                        st.success("Plik audio powiązany z fiszką:")
                        st.audio(st.session_state.flashcard_audio_bytes, format="audio/mp3")
                        if st.button("🗑 Usuń przypisane audio", key="del_fc_audio"):
                            st.session_state.flashcard_audio_bytes = None
                            st.session_state.last_fc_audio_bytes = None
                            # Wymuszenie usunięcia nagrania w widget
                            st.session_state.fc_audio_key = f"fc_audio_{uuid.uuid4()}"
                            st.rerun()
                    
                    if st.button(_("Zapisz gotową fiszkę"), type="primary"):
                        final_defs = st.session_state.p_state['fc_selected_defs'].copy()
                        for line in st.session_state.p_state['fc_custom_defs'].split('\n'):
                            if line.strip(): final_defs.append(line.strip())
                                
                        final_sents = st.session_state.p_state['fc_selected_sents'].copy()
                        for line in st.session_state.p_state['fc_custom_sents'].split('\n'):
                            if line.strip(): final_sents.append(line.strip())

                        async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                            new_rev = WordRevision(
                                user_id=current_user, word=card_word, language=card_lang, base_form=st.session_state.p_state['fc_base_form'],
                                grammar_info=st.session_state.p_state['fc_grammar_info'], definitions=" | ".join(final_defs), inflected_forms=st.session_state.p_state['fc_forms'],
                                example_sentences=" | ".join(final_sents), custom_notes=st.session_state.p_state['fc_custom_notes'],
                                audio_data=st.session_state.flashcard_audio_bytes
                            )
                            session.add(new_rev)
                            await session.commit()
                            
                        st.success(_("Fiszka dodana pomyślnie!"))
                        st.session_state.current_flashcard_draft = False
                        st.session_state.flashcard_audio_bytes = None
                        st.session_state.last_fc_audio_bytes = None
                        
                        # Wyczyść stary widget po dodaniu fiszki
                        st.session_state.fc_audio_key = f"fc_audio_{uuid.uuid4()}"
                        
                        keys_to_clear = ['flashcard_input_word', 'fc_base_form', 'fc_grammar_info', 'fc_forms', 'fc_custom_defs', 'fc_custom_sents', 'fc_custom_notes']
                        for k in keys_to_clear: 
                            st.session_state.p_state[k] = ""
                            if f"wid_{k}" in st.session_state:
                                del st.session_state[f"wid_{k}"]
                                
                        for k in ['fc_selected_defs', 'fc_selected_sents']: 
                            st.session_state.p_state[k] = []
                            if f"wid_{k}" in st.session_state:
                                del st.session_state[f"wid_{k}"]
                                
                        st.session_state.fc_def_options = []
                        st.session_state.fc_sent_options = []
                            
                        time.sleep(0.5)
                        st.rerun()

            elif st.session_state.flashcard_sub_tab == "edit":
                search_query = p_text_input("🔍 " + _("Szukaj fiszki:"), key="search_flashcards")
                
                async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                    res = await session.execute(
                        select(WordRevision).where(WordRevision.user_id == current_user).order_by(WordRevision.creation_date.desc())
                    )
                    all_cards = res.scalars().all()
                    
                    if search_query:
                        q_lower = search_query.lower()
                        all_cards = [
                            c for c in all_cards 
                            if q_lower in c.word.lower() 
                            or (c.definitions and q_lower in c.definitions.lower())
                            or (c.base_form and q_lower in c.base_form.lower())
                        ]
                    
                    if not all_cards: st.info(_("Brak historii aktywności."))
                    
                    for c in all_cards:
                        with st.expander(f"{c.word} ({_(c.language)}) - {_('Następna')}: {c.next_review_date.strftime('%Y-%m-%d')}"):
                            with st.form(key=f"edit_form_{c.id}"):
                                edit_word = st.text_input(_("Słowo"), value=c.word)
                                edit_base = st.text_input(_("Forma podstawowa"), value=c.base_form or "")
                                edit_grammar = st.text_input(_("Info gramatyczne"), value=c.grammar_info or "")
                                edit_defs = st.text_area(_("Definicje"), value=c.definitions or "")
                                edit_forms = st.text_area(_("Odmiany"), value=c.inflected_forms or "")
                                edit_sents = st.text_area(_("Przykłady"), value=c.example_sentences or "")
                                edit_notes = st.text_area(_("Twoje uwagi"), value=c.custom_notes or "")
                                
                                delete_audio = False
                                if c.audio_data:
                                    st.audio(c.audio_data, format="audio/mp3")
                                    delete_audio = st.checkbox(_("Usuń przypisane audio"), key=f"del_aud_{c.id}")

                                col1, col2 = st.columns(2)
                                submitted = col1.form_submit_button(_("💾 Zapisz zmiany"))
                                deleted = col2.form_submit_button(_("🗑 Usuń fiszkę"))
                                
                                if submitted:
                                    c.word = edit_word; c.base_form = edit_base; c.grammar_info = edit_grammar; c.definitions = edit_defs
                                    c.inflected_forms = edit_forms; c.example_sentences = edit_sents; c.custom_notes = edit_notes
                                    if delete_audio: c.audio_data = None
                                    await session.commit()
                                    st.rerun()
                                    
                                if deleted:
                                    await session.delete(c)
                                    await session.commit()
                                    st.rerun()

    elif current_task == "Historia i Zapisane":
        st.header(_("Centrum Danych i Historii"))
        
        t_hist, t_notes = st.tabs([_("Pełna Historia Aktywności"), _("Baza Notatek (Wyszukiwanie Semantyczne)")])
        
        with t_hist:
            async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                res = await session.execute(
                    select(ActivityLog).where(ActivityLog.user_id == current_user).order_by(ActivityLog.created_at.desc()).limit(50)
                )
                logs = res.scalars().all()
                if not logs: st.info(_("Brak historii aktywności."))
                for log in logs:
                    with st.expander(f"{log.created_at.strftime('%Y-%m-%d %H:%M')} | {_(log.task_type.value.upper())}"):
                        st.write(f"**{_('Wejście')}:** {log.input_text}")
                        st.write(f"**{_('Wyjście')}:** {log.output_text}")
        
        with t_notes:
            if not SEMANTIC_SEARCH_AVAILABLE: st.warning(_("Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe."))
            
            q = p_text_input("🔍 " + _("Szukaj w notatkach (po tekście lub znaczeniu):"), key="search_notes")
            
            async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                res = await session.execute(select(Note).where(Note.user_id == current_user).order_by(Note.created_at.desc()))
                notes = res.scalars().all()
                
                if q and SEMANTIC_SEARCH_AVAILABLE:
                    # Generujemy wektor zapytania jako numpy array
                    q_emb = embedder.encode(q)
                    scored_notes = []
                    
                    for n in notes:
                        if n.embedding:
                            try:
                                # Odczytujemy JSON i od razu konwertujemy na wektor matematyczny
                                n_emb = np.array(json.loads(n.embedding))
                                score = cosine_similarity(q_emb, n_emb)
                                scored_notes.append((score, n))
                            except: pass
                            
                    # Sortujemy od najbardziej trafnych
                    scored_notes.sort(key=lambda x: x[0], reverse=True)
                    
                    # --- KLUCZOWA POPRAWKA: Próg zmieniony z 0.1 na 0.35 ---
                    # Poniżej 0.35 to najczęściej przypadkowe powiązania słów
                    notes_to_display = [n for score, n in scored_notes if score > 0.35]
                elif q: notes_to_display = [n for n in notes if n.text and q.lower() in n.text.lower()]
                else: notes_to_display = notes
                
                for n in notes_to_display:
                    with st.container(border=True):
                        st.caption(f"{n.created_at.strftime('%Y-%m-%d %H:%M')} | {_('Typ')}: {_(n.note_type.value)}")
                        if n.text: st.write(n.text)
                        if n.audio_data: st.audio(n.audio_data, format="audio/wav" if n.note_type == NoteType.AUDIO else "audio/mp3")

# ==========================================
# SŁOWNIK TŁUMACZEŃ EKRANU POWITALNEGO (13 JĘZYKÓW)
# ==========================================
WELCOME_SCREEN_TRANSLATIONS = {
    "Angielski": {
        "Twój Osobisty Asystent Językowy AI": "Your Personal AI Language Assistant",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Take foreign language learning to the next level with artificial intelligence. Lingilo combines modern LLM models, spaced repetition algorithms, and semantic search in one place.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Smart Flashcards (SM-2) with AI and Tatoeba database",
        "Korekta Tekstów z wyjaśnieniem błędów": "Text Correction with error explanations",
        "Generowanie Mowy (TTS) i trening wymowy": "Speech Generation (TTS) and pronunciation training",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Advanced Translations preserving context",
        "Baza Notatek z wyszukiwaniem semantycznym": "Knowledge Base with semantic search",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat with a Language Expert in real-time"
    },
    "Niemiecki": {
        "Twój Osobisty Asystent Językowy AI": "Dein persönlicher KI-Sprachassistent",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Bringe das Erlernen von Fremdsprachen durch künstliche Intelligenz auf die nächste Stufe. Lingilo kombiniert moderne LLM-Modelle, Wiederholungsalgorithmen und semantische Suche an einem Ort.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Intelligente Lernkarten (SM-2) mit KI und Tatoeba-Datenbank",
        "Korekta Tekstów z wyjaśnieniem błędów": "Textkorrektur mit Fehlererklärungen",
        "Generowanie Mowy (TTS) i trening wymowy": "Sprachgenerierung (TTS) und Aussprachetraining",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Erweiterte Übersetzungen unter Beibehaltung des Kontexts",
        "Baza Notatek z wyszukiwaniem semantycznym": "Notizdatenbank mit semantischer Suche",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat mit einem Sprachexperten in Echtzeit"
    },
    "Francuski": {
        "Twój Osobisty Asystent Językowy AI": "Votre assistant linguistique personnel IA",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Portez l'apprentissage des langues étrangères au niveau supérieur grâce à l'intelligence artificielle.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Flashcards intelligentes (SM-2) avec IA et base Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Correction de textes avec explications",
        "Generowanie Mowy (TTS) i trening wymowy": "Génération vocale (TTS) et entraînement",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Traductions avancées avec contexte",
        "Baza Notatek z wyszukiwaniem semantycznym": "Base de notes avec recherche sémantique",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Discutez avec un expert linguistique en temps réel"
    },
    "Hiszpański": {
        "Twój Osobisty Asystent Językowy AI": "Tu asistente lingüístico personal de IA",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Lleva el aprendizaje de idiomas al siguiente nivel con inteligencia artificial.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Tarjetas inteligentes (SM-2) con IA y base Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Corrección de textos con explicaciones",
        "Generowanie Mowy (TTS) i trening wymowy": "Generación de voz (TTS) y entrenamiento",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Traducciones avanzadas que preservan el contexto",
        "Baza Notatek z wyszukiwaniem semantycznym": "Base de notas con búsqueda semántica",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat con un experto en idiomas en tiempo real"
    },
    "Włoski": {
        "Twój Osobisty Asystent Językowy AI": "Il tuo assistente linguistico AI",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Porta l'apprendimento delle lingue a un livello superiore con l'intelligenza artificiale.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Flashcard intelligenti (SM-2) con IA e database Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Correzione di testi con spiegazione",
        "Generowanie Mowy (TTS) i trening wymowy": "Generazione vocale (TTS) e allenamento",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Traduzioni avanzate con contesto",
        "Baza Notatek z wyszukiwaniem semantycznym": "Base di note con ricerca semantica",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat con un esperto linguistico in tempo reale"
    },
    "Holenderski": {
        "Twój Osobisty Asystent Językowy AI": "Jouw persoonlijke AI-taalassistent",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Til het leren van vreemde talen naar een hoger niveau met kunstmatige intelligentie.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Slimme flashcards (SM-2) met AI en Tatoeba-database",
        "Korekta Tekstów z wyjaśnieniem błędów": "Tekstcorrectie met foutenanalyse",
        "Generowanie Mowy (TTS) i trening wymowy": "Spraakgeneratie (TTS) en uitspraak",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Geavanceerde vertalingen met behoud van context",
        "Baza Notatek z wyszukiwaniem semantycznym": "Notitiebase met semantisch zoeken",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat met een taalexpert in realtime"
    },
    "Portugalski": {
        "Twój Osobisty Asystent Językowy AI": "Seu assistente de idiomas pessoal de IA",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Eleve o aprendizado de idiomas a um novo patamar com inteligência artificial.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Cartões inteligentes (SM-2) com IA e base Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Correção de textos com explicações",
        "Generowanie Mowy (TTS) i trening wymowy": "Geração de fala (TTS) e treinamento",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Traduções avançadas mantendo o contexto",
        "Baza Notatek z wyszukiwaniem semantycznym": "Base de notas com busca semântica",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat com um especialista em idiomas em tempo real"
    },
    "Rosyjski": {
        "Twój Osobisty Asystent Językowy AI": "Ваш личный ИИ-помощник по языкам",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Поднимите изучение языков на новый уровень с помощью искусственного интеллекта.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Умные карточки (SM-2) с ИИ и базой Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Исправление текста с объяснениями",
        "Generowanie Mowy (TTS) i trening wymowy": "Генерация речи (TTS) и тренировка",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Продвинутый перевод с сохранением контекста",
        "Baza Notatek z wyszukiwaniem semantycznym": "База заметок с семантическим поиском",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Чат с языковым экспертом в реальном времени"
    },
    "Ukraiński": {
        "Twój Osobisty Asystent Językowy AI": "Ваш особистий мовний помічник зі штучним інтелектом",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Підніміть вивчення іноземних мов на новий рівень за допомогою штучного інтелекту.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Розумні картки (SM-2) з ШІ та базою Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Корекція тексту з поясненнями",
        "Generowanie Mowy (TTS) i trening wymowy": "Генерація мовлення (TTS) та тренування",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Розширений переклад зі збереженням контексту",
        "Baza Notatek z wyszukiwaniem semantycznym": "База нотаток із семантичним пошуком",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Чат з мовним експертом у реальному часі"
    },
    "Białoruski": {
        "Twój Osobisty Asystent Językowy AI": "Ваш асабісты моўны памочнік са штучным інтэлектам",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Падніміце вывучэнне замежных моў на новы ўзровень з дапамогай штучнага інтэлекту.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Разумныя карткі (SM-2) з ШІ і базай Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Карэкцыя тэксту z тлумачэннямі",
        "Generowanie Mowy (TTS) i trening wymowy": "Генерацыя маўлення (TTS) і трэніроўка",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Пашыраны пераклад са захаваннем кантэксту",
        "Baza Notatek z wyszukiwaniem semantycznym": "База нататак з семантычным пошукам",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Чат з моўным экспертам у рэальным часе"
    },
    "Słowacki": {
        "Twój Osobisty Asystent Językowy AI": "Váš osobný jazykový asistent AI",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Posuňte štúdium cudzích jazykov na vyššiu úroveň vďaka umelej inteligencii.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Inteligentné kartičky (SM-2) s AI a databázou Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Oprava textov s vysvetlením chýb",
        "Generowanie Mowy (TTS) i trening wymowy": "Generovanie reči (TTS) a tréning výslovnosti",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Pokročilé preklady so zachovaním kontextu",
        "Baza Notatek z wyszukiwaniem semantycznym": "Databáza poznámok so sémantickým vyhľadávaním",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat s jazykovým expertom v reálnom čase"
    },
    "Czeski": {
        "Twój Osobisty Asystent Językowy AI": "Váš osobní jazykový asistent AI",
        "Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu.": "Posuňte studium cizích jazyků na vyšší úroveň díky umělé inteligenci.",
        "Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba": "Chytré kartičky (SM-2) s AI a databází Tatoeba",
        "Korekta Tekstów z wyjaśnieniem błędów": "Oprava textů s vysvětlením chyb",
        "Generowanie Mowy (TTS) i trening wymowy": "Generování řeči (TTS) a trénink výslovnosti",
        "Zaawansowane Tłumaczenia z zachowaniem kontekstu": "Pokročilé překlady se zachováním kontextu",
        "Baza Notatek z wyszukiwaniem semantycznym": "Databáze poznámok se sémantickým vyhledáváním",
        "Czat z Ekspertem językowym w czasie rzeczywistym": "Chat s jazykovým expertom v reálném čase"
    }
}

# Automatyczne dołączenie słownika powitalnego do głównego słownika STATIC_TRANSLATIONS
for lang, translations in WELCOME_SCREEN_TRANSLATIONS.items():
    if lang not in STATIC_TRANSLATIONS:
        STATIC_TRANSLATIONS[lang] = {}
    STATIC_TRANSLATIONS[lang].update(translations)

# ==========================================
# FUNKCJA MAIN Z BEZPIECZNĄ OBSŁUGĄ COOKIES I OAUTH
# ==========================================
import streamlit as st
import requests
import asyncio
import time

# Gotowy, społecznościowy wzorzec bezpiecznego pobierania ciasteczek (Boilerplate)
def get_all_cookies_secure(cookie_manager):
    # Społeczność Streamlita zaleca małą pętlę i timeout, by dać czas komponentowi JS
    for _ in range(15):  # maksymalnie 15 prób (około 1.5 - 2 sekund oczekiwania)
        cookies = cookie_manager.get_all()
        if cookies is not None:
            return cookies
        time.sleep(0.1)  # Czekamy 100ms na asynchroniczną odpowiedź z frontendu
    return {}  # Zwróć pusty słownik, jeśli po 1.5s nic nie przyszło

# ==========================================
# OFICJALNY, WBUDOWANY STANDARD MAIN (ST.CONTEXT)
# ==========================================
def main():
    st.set_page_config(page_title=_("Lingilo - Twój pomocnik językowy"), layout="wide")

    if "logged_out_flag" not in st.session_state:
        st.session_state.logged_out_flag = False

    is_auth0_callback = "code" in st.query_params

    # --- NATYWNE I BŁYSKAWICZNE ODZYSKIWANIE SESJI (Działa lokalnie i w chmurze) ---
    if "token" not in st.session_state and not is_auth0_callback:
        if not st.session_state.get("logged_out_flag"):
            try:
                # Oficjalne pobranie ciasteczek za pomocą wbudowanego interfejsu Streamlit
                saved_token = st.context.cookies.get("auth0_token")
                
                if saved_token and str(saved_token).strip() != "" and saved_token != "None":
                    st.session_state.token = {"access_token": saved_token}
                    st.rerun()
            except Exception:
                pass

    # --- KONTROLA WIDOKU ---
    show_main_app = False

    if "token" not in st.session_state:
        # POKAZUJEMY EKRAN LOGOWANIA (Zgłoszony priorytetowo - ma gwarancję wyświetlenia)
        st.title("🌍 Lingilo")
        st.subheader(_("Twój Osobisty Asystent Językowy AI"))
        
        st.markdown(_("Wznieś naukę języków obcych na wyższy poziom dzięki sztucznej inteligencji. Lingilo łączy nowoczesne modele LLM, algorytmy powtórek i wyszukiwanie semantyczne w jednym miejscu."))
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(
                f"* 📚 **{_('Inteligentne Fiszki (SM-2) z AI i bazą Tatoeba')}**\n"
                f"* ✍️ **{_('Korekta Tekstów z wyjaśnieniem błędów')}**\n"
                f"* 🗣️ **{_('Generowanie Mowy (TTS) i trening wymowy')}**"
            )
        with col_f2:
            st.markdown(
                f"* 🌐 **{_('Zaawansowane Tłumaczenia z zachowaniem kontekstu')}**\n"
                f"* 🧠 **{_('Baza Notatek z wyszukiwaniem semantycznym')}**\n"
                f"* 💬 **{_('Czat z Ekspertem językowym w czasie rzeczywistym')}**"
            )
            
        st.markdown("---")
        
        ui_langs_display = [NATIVE_LANGUAGE_NAMES[l] for l in SUPPORTED_LANGUAGES]
        sel_ui_trans = st.selectbox(
            _("Język interfejsu aplikacji:"), 
            ui_langs_display, 
            index=SUPPORTED_LANGUAGES.index(st.session_state.ui_language),
            key="login_language_picker"
        )
        new_ui_lang = SUPPORTED_LANGUAGES[ui_langs_display.index(sel_ui_trans)]
        
        if new_ui_lang != st.session_state.ui_language:
            st.session_state.ui_language = new_ui_lang
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True) 
        
        # PRZYCISK LOGOWANIA (Zawsze stabilny na ekranie, czysty HTML/JS)
        result = oauth2.authorize_button(
            name=_("Zaloguj przez Auth0"),
            icon="https://auth0.com",
            redirect_uri=REDIRECT_URI,  
            scope="openid email profile"
        )
        
        if result and "token" in result:
            token_data = result.get("token")
            st.session_state.token = token_data
            st.session_state.logged_out_flag = False
            
            # Zapis ciasteczka realizujemy przez dedykowanego menedżera tylko po kliknięciu
            cookie_manager_save = stx.CookieManager(key="lingilo_global_cookie_manager")
            access_tok = token_data.get("access_token") if isinstance(token_data, dict) else token_data
            try:
                cookie_manager_save.set("auth0_token", access_tok, key="set_cookie_on_login")
                time.sleep(0.2)
            except Exception:
                pass
            st.rerun()
            
    else:
        show_main_app = True

    # --- RENDEROWANIE GŁÓWNEJ APLIKACJI (Po poprawnym zalogowaniu) ---
    if show_main_app:
        cookie_manager = stx.CookieManager(key="lingilo_global_cookie_manager")
        
        # OBSŁUGA FORCE_LOGOUT
        if st.session_state.get("force_logout"):
            try:
                cookie_manager.set("auth0_token", "", key="del_auth_cookie_force")
                time.sleep(0.2)
            except Exception:
                pass
            st.session_state.pop("token", None)
            st.session_state.pop("user_info", None)
            st.session_state.force_logout = False
            st.session_state.logged_out_flag = True
            st.query_params.clear()
            st.rerun()

        token_data = st.session_state.token
        access_token_str = token_data.get("access_token") if isinstance(token_data, dict) else token_data

        if "user_info" not in st.session_state:
            try:
                headers_auth = {"Authorization": f"Bearer {access_token_str}"}
                response = requests.get(f"https://{AUTH0_DOMAIN}/userinfo", headers=headers_auth)
                response.raise_for_status()
                st.session_state.user_info = response.json()
            except Exception as e:
                st.warning("Sesja Auth0 wygasła lub wystąpił błąd. Trwa wylogowywanie...")
                try:
                    cookie_manager.set("auth0_token", "", key="del_expired_session")
                    time.sleep(0.2)
                except Exception:
                    pass
                
                st.session_state.pop("token", None)
                st.session_state.pop("user_info", None)
                st.session_state.logged_out_flag = True 
                time.sleep(1.0)
                st.query_params.clear()
                st.rerun()

        if "user_info" in st.session_state:
            current_user = st.session_state.user_info.get("sub", "default_user")
            st.session_state.current_user_id = current_user

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_core(current_user, cookie_manager))
            finally:
                loop.close()

if __name__ == "__main__":
    main()