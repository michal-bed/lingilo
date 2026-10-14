import sys
import os
import io
import re
import json
import asyncio
import time
from datetime import datetime, timedelta

import openai
import instructor
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

import nest_asyncio
nest_asyncio.apply()

# --- NAPRAWA BŁĘDÓW SSL I ASYNCIO W SYSTEMIE WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ['SSL_CERT_FILE'] = certifi.where()


# ==========================================
# KONFIGURACJA I STAN SESJI
# ==========================================
env = dotenv_values(".env")
load_dotenv()

# Bezpieczne pobieranie kluczy (omija błąd braku pliku secrets lokalnie)
def get_secret(key, default_val=""):
    if key in env and env[key]: 
        return env[key]
    if os.getenv(key): 
        return os.getenv(key)
    try: 
        return st.secrets[key]
    except Exception: 
        return default_val

# ==========================================
# KONFIGURACJA AUTH0 (OAUTH2)
# ==========================================
AUTH0_DOMAIN = get_secret("AUTH0_DOMAIN")  
CLIENT_ID = get_secret("CLIENT_ID")         
CLIENT_SECRET = get_secret("CLIENT_SECRET") 

oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_endpoint=f"https://{AUTH0_DOMAIN}/authorize",
    token_endpoint=f"https://{AUTH0_DOMAIN}/oauth/token",
    refresh_token_endpoint=f"https://{AUTH0_DOMAIN}/oauth/token",
    revoke_token_endpoint=f"https://{AUTH0_DOMAIN}/oauth/revoke"
)

# ==========================================
# WYSZUKIWANIE SEMANTYCZNE (PAMIĘĆ PODRĘCZNA)
# ==========================================
@st.cache_resource(show_spinner=True)
def load_embedder():
    try:
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except ImportError:
        return None

embedder = load_embedder()
SEMANTIC_SEARCH_AVAILABLE = embedder is not None


SUPPORTED_LANGUAGES = [
    "Polski", "Angielski", "Niemiecki", "Holenderski", "Francuski", "Hiszpański", "Włoski", 
    "Portugalski", "Rosyjski", "Ukraiński", "Białoruski", "Słowacki", "Czeski"
]

# ==========================================
# SCHEMATY AI (INSTRUCTOR & PYDANTIC)
# ==========================================
class GrammarFeatures(BaseModel):
    part_of_speech: str = Field(description="Część mowy (np. rzeczownik, czasownik, przymiotnik, przysłówek, zaimek)")
    gender: Optional[str] = Field(None, description="Rodzaj gramatyczny. Wypełnij tylko jeśli dotyczy danej części mowy (pomiń dla przysłówków!).")
    number: Optional[str] = Field(None, description="Liczba (np. pojedyncza, mnoga).")
    case: Optional[str] = Field(None, description="Przypadek (np. mianownik). Tylko dla deklinacji.")
    person: Optional[str] = Field(None, description="Osoba. Tylko dla czasowników i zaimków osobowych.")
    tense: Optional[str] = Field(None, description="Czas (np. teraźniejszy, Past Simple). Tylko dla czasowników.")
    aspect: Optional[str] = Field(None, description="Aspekt (np. dokonany). Tylko dla czasowników.")
    degree: Optional[str] = Field(None, description="Stopień. Tylko dla przymiotników i przysłówków.")

class ExampleSentence(BaseModel):
    original: str = Field(description="Przykładowe zdanie w docelowym języku obcym.")
    translation: str = Field(description="Tłumaczenie tego zdania na język bazowy użytkownika.")

class FlashcardData(BaseModel):
    base_form: str = Field(description="Forma podstawowa słowa")
    grammar_info: GrammarFeatures
    definitions: List[str] = Field(description="Krótkie definicje lub tłumaczenia (maksymalnie 3)")
    forms: List[str] = Field(description="Najważniejsze formy odmienione. Maksymalnie 3.")
    sentences: List[ExampleSentence] = Field(description="Przykładowe zdania z użyciem tego słowa (od 2 do 4 zdań)")

# ==========================================
# WBUDOWANY SŁOWNIK INTERFEJSU
# ==========================================
STATIC_TRANSLATIONS = {
    "Angielski": {
        "Polski": "Polish", "Angielski": "English", "Niemiecki": "German", "Holenderski": "Dutch", "Francuski": "French", "Hiszpański": "Spanish", "Włoski": "Italian", "Portugalski": "Portuguese", "Rosyjski": "Russian", "Ukraiński": "Ukrainian", "Białoruski": "Belarusian", "Słowacki": "Slovak", "Czeski": "Czech",
        "Ustawienia": "Settings", "Język interfejsu aplikacji:": "App interface language:", "Dostawca modelu AI:": "AI model provider:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Paid option is temporarily unavailable. Using free Groq model.", "Wybierz zadanie:": "Select task:",
        "Groq (Darmowe chmurowe)": "Groq (Free Cloud)", "OpenAI (Płatne)": "OpenAI (Paid)",
        "Tłumaczenie": "Translation", "Korekta tekstu": "Text correction", "Generowanie audio": "Audio generation", "Zapytanie": "Query", "Notatki": "Notes", "Powtórki": "Flashcards", "Historia i Zapisane": "History & Saved",
        "Wprowadź tekst do przetłumaczenia": "Enter text to translate", "Twój tekst:": "Your text:", "Język docelowy": "Target language", "Tłumacz": "Translate", "Trwa tłumaczenie...": "Translating...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "You are a translator. Return only the clean translation.",
        "Korekta błędów i redakcja": "Error correction and editing", "Tekst do poprawy:": "Text to correct:", "Popraw tekst": "Correct text", "Analiza błędów...": "Analyzing errors.", "Popraw błędy. Zwróć poprawiony tekst.": "Correct errors. Return corrected text.",
        "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Then explain the changes made in bullet points and briefly justify why the new version is better.",
        "Zamiana tekstu na mowę": "Text to speech", "Wpisz tekst:": "Enter text:", "Generuj wymowę": "Generate speech", "Przygotowywanie...": "Preparing...", "Zapisz to Audio do Historii": "Save Audio to History", "Wygenerowane TTS": "Generated TTS", "Zapisano audio w historii!": "Audio saved in history!",
        "Czat z ekspertem językowym": "Chat with language expert", "Zadaj pytanie:": "Ask a question:", "Zapytaj": "Ask", "Piszę odpowiedź...": "Writing response...", "Jesteś ekspertem językowym.": "You are a language expert.",
        "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Save speech and text with semantic search", "🎤 Nagraj notatkę głosową": "🎤 Record a voice note", "Opcjonalnie: Transkrybuj mowę": "Optional: Transcribe speech", "Rozpoznawanie mowy...": "Recognizing speech...", "Błąd transkrypcji": "Transcription error", "Treść notatki tekstowej (lub transkrypcji):": "Text note content (or transcription):", "Zapisz Notatkę do bazy": "Save Note to database", "Notatka jest pusta!": "Note is empty!", "Zapisywanie i generowanie wektorów...": "Saving and generating vectors...", "Notatka zapisana pomyślnie!": "Note saved successfully!",
        "Twój system nauki (Spaced Repetition)": "Your learning system (Spaced Repetition)", "Nauka (SM-2)": "Study (SM-2)", "Dodaj Fiszki": "Add Flashcards", "Edytuj / Usuń": "Edit / Delete", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Congratulations! No words for today.", "Do powtórzenia": "To review", "Język": "Language", "Baza": "Base", "Gramatyka": "Grammar", "Pokaż odpowiedź": "Show answer", "Definicje": "Definitions", "Odmiany": "Inflections", "Przykłady": "Examples", "Nie pamiętam": "Don't remember", "Z trudem": "Hard", "Dobrze": "Good", "Z łatwością": "Easy",
        "Kreator z analizą lingwistyczną AI": "Creator with AI linguistic analysis", "Wpisz słowo lub wyrażenie:": "Enter word or expression:", "Język słowa:": "Word language:", "Pobierz dane przez AI": "Fetch data via AI", "AI analizuje słowo jako API...": "AI is analyzing the word...", "Pobierz zdania z bazy Tatoeba": "Fetch sentences from Tatoeba", "Szukam autentycznych przykładów...": "Searching for authentic examples...", "API nie znalazło przykładów.": "API found no examples.", "Wprowadź ręcznie": "Enter manually", "Forma podstawowa": "Base form", "Info gramatyczne": "Grammar info", "Wybierz definicje do zapisania:": "Select definitions to save:", "Wybierz przykłady zdań:": "Select example sentences:", "Zapisz gotową fiszkę": "Save flashcard", "Fiszka dodana pomyślnie!": "Flashcard added successfully!",
        "Twoje własne uwagi (opcjonalnie):": "Your custom notes (optional):", "Twoje uwagi": "Your notes",
        "Język tłumaczenia (definicje):": "Translation language (definitions):", "Słowo": "Word", "💾 Zapisz zmiany": "💾 Save changes",
        "Audio (opcjonalnie):": "Audio (optional):", "Generuj wymowę (TTS)": "Generate pronunciation (TTS)", "🎤 Nagraj wymowę": "🎤 Record pronunciation", "Usuń przypisane audio": "Delete assigned audio",
        "Następna": "Next", "🗑 Usuń fiszkę": "🗑 Delete flashcard", "Centrum Danych i Historii": "Data and History Center", "Pełna Historia Aktywności": "Full Activity History", "Baza Notatek (Wyszukiwanie Semantyczne)": "Notes Base (Semantic Search)", "Brak historii aktywności.": "No activity history.", "Wejście": "Input", "Wyjście": "Output", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Semantic search inactive. Text search works.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Search notes (by text or meaning):", "Typ": "Type",
        "📝 Zapisz jako notatkę": "📝 Save as note", "📇 Przekaż do fiszek": "📇 Send to flashcards", "🔊 Przekaż do TTS": "🔊 Send to TTS", "Zapisano w notatkach!": "Saved to notes!"
    },
    "Niemiecki": {
        "Polski": "Polnisch", "Angielski": "Englisch", "Niemiecki": "Deutsch", "Holenderski": "Niederländisch", "Francuski": "Französisch", "Hiszpański": "Spanisch", "Włoski": "Italienisch", "Portugalski": "Portugiesisch", "Rosyjski": "Russisch", "Ukraiński": "Ukrainisch", "Białoruski": "Weißrussisch", "Słowacki": "Slowakisch", "Czeski": "Tschechisch",
        "Ustawienia": "Einstellungen", "Język interfejsu aplikacji:": "App-Sprache:", "Dostawca modelu AI:": "KI-Modellanbieter:", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Bezahlte Option nicht verfügbar. Nutze kostenloses Groq-Modell.", "Wybierz zadanie:": "Aufgabe wählen:",
        "Groq (Darmowe chmurowe)": "Groq (Kostenlose Cloud)", "OpenAI (Płatne)": "OpenAI (Kostenpflichtig)",
        "Tłumaczenie": "Übersetzung", "Korekta tekstu": "Textkorrektur", "Generowanie audio": "Audio-Generierung", "Zapytanie": "Abfrage", "Notatki": "Notizen", "Powtórki": "Lernkarten", "Historia i Zapisane": "Verlauf & Gespeichert",
        "Wprowadź tekst do przetłumaczenia": "Text zum Übersetzen eingeben", "Twój tekst:": "Dein Text:", "Język docelowy": "Zielsprache", "Tłumacz": "Übersetzen", "Trwa tłumaczenie...": "Übersetzung läuft...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Du bist Übersetzer. Gib nur die reine Übersetzung zurück.",
        "Korekta błędów i redakcja": "Fehlerkorrektur und Bearbeitung", "Tekst do poprawy:": "Zu korrigierender Text:", "Popraw tekst": "Text korrigieren", "Analiza błędów...": "Fehler analysieren...", "Popraw błędy. Zwróć poprawiony tekst.": "Fehler korrigieren. Korrigierten Text zurückgeben.",
        "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Erkläre dann die Änderungen in Stichpunkten und begründe kurz, warum die neue Version besser ist.",
        "Zamiana tekstu na mowę": "Text-zu-Sprache", "Wpisz tekst:": "Text eingeben:", "Generuj wymowę": "Aussprache generieren", "Przygotowywanie...": "Vorbereiten...", "Zapisz to Audio do Historii": "Audio im Verlauf speichern", "Wygenerowane TTS": "Generiertes TTS", "Zapisano audio w historii!": "Audio im Verlauf gespeichert!",
        "Czat z ekspertem językowym": "Chat mit Sprachexperten", "Zadaj pytanie:": "Frage stellen:", "Zapytaj": "Fragen", "Piszę odpowiedź...": "Antwort wird geschrieben...", "Jesteś ekspertem językowym.": "Du bist ein Sprachexperte.",
        "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Sprache und Text mit semantischer Suche speichern", "🎤 Nagraj notatkę głosową": "🎤 Sprachnotiz aufnehmen", "Opcjonalnie: Transkrybuj mowę": "Optional: Sprache transkribieren", "Rozpoznawanie mowy...": "Spracherkennung...", "Błąd transkrypcji": "Transkriptionsfehler", "Treść notatki tekstowej (lub transkrypcji):": "Inhalt der Textnotiz:", "Zapisz Notatkę do bazy": "Notiz speichern", "Notatka jest pusta!": "Notiz ist leer!", "Zapisywanie i generowanie vektorów...": "Speichern und Vektoren generieren...", "Notatka zapisana pomyślnie!": "Notiz erfolgreich gespeichert!",
        "Twój system nauki (Spaced Repetition)": "Dein Lernsystem (Spaced Repetition)", "Nauka (SM-2)": "Lernen (SM-2)", "Dodaj Fiszki": "Lernkarten hinzufügen", "Edytuj / Usuń": "Bearbeiten / Löschen", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Glückwunsch! Keine Wörter für heute.",
        "Do powtórzenia": "Zu wiederholen", "Język": "Sprache", "Baza": "Basis", "Gramatyka": "Grammatik", "Pokaż odpowiedź": "Antwort anzeigen", "Definicje": "Definitionen", "Odmiany": "Beugungen", "Przykłady": "Beispiele", "Nie pamiętam": "Nicht erinnert", "Z trudem": "Schwer", "Dobrze": "Gut", "Z łatwością": "Einfach",
        "Kreator z analizą lingwistyczną AI": "Ersteller mit KI-Linguistikanalyse", "Wpisz słowo lub wyrażenie:": "Wort oder Ausdruck eingeben:", "Język słowa:": "Wortsprache:", "Pobierz dane przez AI": "Daten über KI abrufen", "AI analizuje słowo jako API...": "KI analysiert das Wort...", "Pobierz zdania z bazy Tatoeba": "Sätze von Tatoeba abrufen", "Szukam autentycznych przykładów...": "Suche nach Beispielen...", "API nie znalazło przykładów.": "API hat keine Beispiele gefunden.",
        "Wprowadź ręcznie": "Manuell eingeben", "Forma podstawowa": "Grundform", "Info gramatyczne": "Grammatik-Info", "Wybierz definicje do zapisania:": "Definitionen zum Speichern auswählen:", "Wybierz przykłady zdań:": "Beispielsätze auswählen:", "Zapisz gotową fiszkę": "Lernkarte speichern", "Fiszka dodana pomyślnie!": "Lernkarte erfolgreich hinzugefügt!",
        "Twoje własne uwagi (opcjonalnie):": "Deine eigenen Notizen (optional):", "Twoje uwagi": "Deine Notizen",
        "Język tłumaczenia (definicje):": "Übersetzungssprache (Definitionen):", "Słowo": "Wort", "💾 Zapisz zmiany": "💾 Änderungen speichern",
        "Audio (opcjonalnie):": "Audio (optional):", "Generuj wymowę (TTS)": "Aussprache generieren (TTS)", "🎤 Nagraj wymowę": "🎤 Aussprache aufnehmen", "Usuń przypisane audio": "Zugewiesenes Audio löschen",
        "Następna": "Nächste", "🗑 Usuń fiszkę": "🗑 Lernkarte löschen", "Centrum Danych i Historii": "Daten- und Verlaufszentrum", "Pełna Historia Aktywności": "Vollständiger Aktivitätsverlauf", "Baza Notatek (Wyszukiwanie Semantyczne)": "Notizdatenbank (Semantische Suche)", "Brak historii aktywności.": "Kein Aktivitätsverlauf.",
        "Wejście": "Eingabe", "Wyjście": "Ausgabe", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Semantische Suche inaktiv. Textsuche funktioniert.", "Szukaj w notatkach (po tekście lub znaczeniu):": "In Notizen suchen:", "Typ": "Typ",
        "📝 Zapisz jako notatkę": "📝 Als Notiz speichern", "📇 Przekaż do fiszek": "📇 Zu Lernkarten hinzufügen", "🔊 Przekaż do TTS": "🔊 An TTS senden", "Zapisano w notatkach!": "In Notizen gespeichert!"
    },
    "Francuski": {
        "Polski": "Polonais", "Angielski": "Anglais", "Niemiecki": "Allemand", "Holenderski": "Néerlandais", "Francuski": "Français", "Hiszpański": "Espagnol", "Włoski": "Italien", "Portugalski": "Portugais", "Rosyjski": "Russe", "Ukraiński": "Ukrainien", "Białoruski": "Biélorusse", "Słowacki": "Slovaque", "Czeski": "Tchèque",
        "Ustawienia": "Paramètres", "Język interfejsu aplikacji:": "Langue de l'interface :", "Dostawca modelu AI:": "Fournisseur d'IA :", "Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq.": "Option payante indisponible. Utilisation de Groq gratuit.", "Wybierz zadanie:": "Sélectionner une tâche :",
        "Groq (Darmowe chmurowe)": "Groq (Cloud Gratuit)", "OpenAI (Płatne)": "OpenAI (Payant)",
        "Tłumaczenie": "Traduction", "Korekta tekstu": "Correction de texte", "Generowanie audio": "Génération audio", "Zapytanie": "Requête", "Notatki": "Notes", "Powtórki": "Révisions", "Historia i Zapisane": "Historique & Enregistré",
        "Wprowadź tekst do przetłumaczenia": "Entrez le texte à traduire", "Twój tekst:": "Votre texte :", "Język docelowy": "Langue cible", "Tłumacz": "Traduire", "Trwa tłumaczenie...": "Traduction en cours...", "Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie.": "Vous êtes traducteur. Retournez uniquement la traduction.",
        "Korekta błędów i redakcja": "Correction d'erreurs et édition", "Tekst do poprawy:": "Texte à corriger :", "Popraw tekst": "Corriger le texte", "Analiza błędów...": "Analyse des erreurs...", "Popraw błędy. Zwróć poprawiony tekst.": "Corrigez les erreurs. Retournez le texte corrigé.",
        "Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.": "Expliquez ensuite les modifications en points et justifiez pourquoi la nouvelle version est meilleure.",
        "Zamiana tekstu na mowę": "Synthèse vocale", "Wpisz tekst:": "Entrez le texte :", "Generuj wymowę": "Générer la voix", "Przygotowywanie...": "Préparation...", "Zapisz to Audio do Historii": "Enregistrer l'audio dans l'historique", "Wygenerowane TTS": "TTS Généré", "Zapisano audio w historii!": "Audio enregistré !",
        "Czat z ekspertem językowym": "Discutez avec un expert", "Zadaj pytanie:": "Posez une question :", "Zapytaj": "Demander", "Piszę odpowiedź...": "Rédaction de la réponse...", "Jesteś expert linguistique.": "Vous êtes un expert linguistique.",
        "Zapisuj mowę i tekst z wyszukiwaniem semantycznym": "Enregistrez avec recherche sémantique", "🎤 Nagraj notatkę głosową": "🎤 Enregistrer une note vocale", "Opcjonalnie: Transkrybuj mowę": "Optionnel : Transcrire la voix", "Rozpoznawanie mowy...": "Reconnaissance vocale...", "Błąd transkrypcji": "Erreur de transcription", "Treść notatki tekstowej (lub transkrypcji):": "Contenu de la note :", "Zapisz Notatkę do bazy": "Enregistrer la note", "Notatka jest pusta!": "La note est vide !", "Zapisywanie i generowanie vektorów...": "Enregistrement en cours...", "Notatka zapisana pomyślnie!": "Note enregistrée avec succès !",
        "Twój system nauki (Spaced Repetition)": "Votre système d'apprentissage", "Nauka (SM-2)": "Étudier (SM-2)", "Dodaj Fiszki": "Ajouter Flashcards", "Edytuj / Usuń": "Modifier / Supprimer", "🎉 Gratulacje! Brak słówek na dziś.": "🎉 Félicitations ! Aucun mot pour aujourd'hui.", "Do powtórzenia": "À réviser", "Język": "Langue", "Baza": "Base", "Gramatyka": "Grammaire", "Pokaż odpowiedź": "Afficher la réponse", "Definicje": "Définitions", "Odmiany": "Déclinaisons", "Przykłady": "Exemples", "Nie pamiętam": "Oublié", "Z trudem": "Difficile", "Dobrze": "Bien", "Z łatwością": "Facile",
        "Kreator z analizą lingwistyczną AI": "Créateur avec analyse IA", "Wpisz słowo lub wyrażenie:": "Entrez un mot :", "Język słowa:": "Langue du mot :", "Pobierz dane przez AI": "Obtenir les données IA", "AI analizuje słowo jako API...": "L'IA analyse le mot...", "Pobierz zdania z bazy Tatoeba": "Récupérer des phrases de Tatoeba", "Szukam autentycznych przykładów...": "Recherche d'exemples...", "API nie znalazło przykładów.": "L'API n'a rien trouvé.",
        "Wprowadź ręcznie": "Saisir manuellement", "Forma podstawowa": "Forme de base", "Info gramatyczne": "Infos grammaticales", "Wybierz definicje do zapisania:": "Sélectionnez les définitions :", "Wybierz przykłady zdań:": "Sélectionnez les phrases :", "Zapisz gotową fiszkę": "Enregistrer la flashcard", "Fiszka dodana pomyślnie!": "Flashcard ajoutée !",
        "Twoje własne uwagi (opcjonalnie):": "Vos propres notes (facultatif):", "Twoje uwagi": "Vos notes",
        "Język tłumaczenia (definicje):": "Langue de traduction (définitions) :", "Słowo": "Mot", "💾 Zapisz zmiany": "💾 Enregistrer les modifications",
        "Audio (opcjonalnie):": "Audio (facultatif) :", "Generuj wymowę (TTS)": "Générer la prononciation (TTS)", "🎤 Nagraj wymowę": "🎤 Enregistrer la prononciation", "Usuń przypisane audio": "Supprimer l'audio assigné",
        "Następna": "Prochaine", "🗑 Usuń fiszkę": "🗑 Supprimer la flashcard", "Centrum Danych i Historii": "Centre de Données", "Pełna Historia Aktywności": "Historique complet", "Baza Notatek (Wyszukiwanie Semantyczne)": "Base de notes (Recherche Sémantique)", "Brak historii aktywności.": "Aucun historique.",
        "Wejście": "Entrée", "Wyjście": "Sortie", "Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe.": "Recherche sémantique inactive.", "Szukaj w notatkach (po tekście lub znaczeniu):": "Rechercher dans les notes :", "Typ": "Type",
        "📝 Zapisz jako notatkę": "📝 Enregistrer comme note", "📇 Przekaż do fiszek": "📇 Envoyer aux flashcards", "🔊 Przekaż do TTS": "🔊 Envoyer à la synthèse vocale", "Zapisano w notatkach!": "Enregistré dans les notes !"
    }
}

def init_session_state():
    persistent_keys = {
        'trans_text': "", 'corr_text': "", 'tts_text': "", 'inq_text': "",
        'custom_defs': "", 'custom_sents': "", 'custom_notes': "",
        'ui_language': "Polski", 'translation_target_language': "Angielski",
        'ai_provider': "Groq (Darmowe chmurowe)",
        'active_task': "Tłumaczenie",
        'flashcard_input_word': "",
        'current_user_id': None,
        'prefs_loaded': False,
        'last_saved_drafts': None
    }
    for k, v in persistent_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

    defaults = {
        'translated_text': "", 'corrected_text': "", 'inquiry_response': "",
        'note_audio_bytes': None, 'note_audio_text': "",
        'audio_generation_audio_file': None,
        'show_flashcard_answer': False, 'current_flashcard_draft': None,
        'flashcard_audio_bytes': None 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def _(text): 
    lang = st.session_state.ui_language
    if lang == "Polski" or not text.strip():
        return text
    return STATIC_TRANSLATIONS.get(lang, STATIC_TRANSLATIONS.get("Angielski", {})).get(text, text)

# ==========================================
# BAZA DANYCH (NEON / POSTGRESQL)
# ==========================================
def get_engine():
    db_url = get_secret("DATABASE_URL", "sqlite+aiosqlite:///lingilo_local.db")
    
    if "?" in db_url:
        db_url = db_url.split("?")[0]
    
    connect_args = {"ssl": True} if "postgresql" in db_url else {}
    
    return create_async_engine(
        db_url, 
        echo=False, 
        connect_args=connect_args,
        poolclass=NullPool
    )

Base = declarative_base()

class ModelType(enum.Enum):
    GPT_4O = "gpt-4o"
    LLAMA_3 = "llama3"

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
    
    # Krok 1: Tworzymy wszystkie tabele w pierwszej transakcji
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Krok 2: Próbujemy dodać nową kolumnę w odrębnej transakcji.
    # Jeśli kolumna już istnieje (lub tabela jest nowa), 
    # błąd zostanie zignorowany i nie cofnie utworzenia tabel z Kroku 1.
    try:
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE user_preferences ADD COLUMN session_drafts TEXT DEFAULT '{}'"))
    except Exception:
        pass

# ==========================================
# POMOCNICZE FUNKCJE BAZY DANYCH
# ==========================================
async def load_user_preferences(user_id):
    # Dodajemy expire_on_commit=False, aby obiekt "przeżył" zamknięcie sesji
    async_session = sessionmaker(
        bind=get_engine(), 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        res = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = res.scalars().first()
        
        if not prefs:
            prefs = UserPreferences(user_id=user_id)
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

# --- NOWA FUNKCJA ZAPISUJĄCA WERSJE ROBOCZE DO BAZY ---
async def save_current_drafts(current_user):
    if not current_user or not st.session_state.get('prefs_loaded'):
        return
        
    drafts_to_save = {
        'trans_text': st.session_state.get('trans_text', ''),
        'translated_text': st.session_state.get('translated_text', ''),
        'corr_text': st.session_state.get('corr_text', ''),
        'corrected_text': st.session_state.get('corrected_text', ''),
        'tts_text': st.session_state.get('tts_text', ''),
        'inq_text': st.session_state.get('inq_text', ''),
        'inquiry_response': st.session_state.get('inquiry_response', ''),
        'flashcard_input_word': st.session_state.get('flashcard_input_word', ''),
        'custom_defs': st.session_state.get('custom_defs', ''),
        'custom_sents': st.session_state.get('custom_sents', ''),
        'custom_notes': st.session_state.get('custom_notes', '')
    }
    current_drafts_json = json.dumps(drafts_to_save)
    
    if current_drafts_json != st.session_state.get('last_saved_drafts'):
        await update_user_preference(current_user, "session_drafts", current_drafts_json)
        st.session_state.last_saved_drafts = current_drafts_json

# ==========================================
# ALGORYTM SUPERMEMO-2 (SM-2)
# ==========================================
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

# ==========================================
# AI KLIENT Z INSTRUCTOREM
# ==========================================
def get_async_openai_client():
    api_key = get_secret("GROQ_API_KEY", "BRAK_KLUCZA")
    return openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

def get_async_instructor_client():
    api_key = get_secret("GROQ_API_KEY", "BRAK_KLUCZA")
    base_client = openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)

async def monitor_llm(system_prompt, user_prompt, task_type: TaskType, return_json=False):
    client = get_async_openai_client()
    try:
        kwargs = {"response_format": {"type": "json_object"}} if return_json else {}
        chat = await client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="openai/gpt-oss-20b", **kwargs
        )
        output = chat.choices[0].message.content
        
        async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
            session.add(ActivityLog(user_id=st.session_state.current_user_id, task_type=task_type, input_text=user_prompt, output_text=output))
            await session.commit()
            
        return output
    except Exception as e:
        return f"Błąd AI: {e}"

async def generate_word_data(word, language, target_language):
    client = get_async_instructor_client()
    system_prompt = f"Jesteś wybitnym lingwistą. Tworzysz fiszkę dla słowa w języku '{language}'. Pamiętaj, aby definicje oraz tłumaczenia zdań (translation) były napisane w języku '{target_language}'."
    
    try:
        flashcard = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            response_model=FlashcardData,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analizuj: '{word}' w języku {language}."}
            ]
        )
        
        g = flashcard.grammar_info
        features = [f for f in [g.gender, g.number, g.case, g.person, g.tense, g.aspect, g.degree] if f]
        
        if features:
            formatted_grammar = f"{g.part_of_speech} ({', '.join(features)})"
        else:
            formatted_grammar = g.part_of_speech
            
        formatted_sentences = [f"{s.original} ➔ {s.translation}" for s in flashcard.sentences]
        
        async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
            session.add(ActivityLog(user_id=st.session_state.current_user_id, task_type=TaskType.REVISION, input_text=f"Analizuj: {word} w języku {language}", output_text=flashcard.model_dump_json()))
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

TATOEBA_LANG_MAP = {
    "Polski": "pol", "Angielski": "eng", "Niemiecki": "deu", 
    "Holenderski": "nld", "Francuski": "fra", "Hiszpański": "spa", 
    "Włoski": "ita", "Portugalski": "por", "Rosyjski": "rus", 
    "Ukraiński": "ukr", "Białoruski": "bel", "Słowacki": "slk", "Czeski": "ces"
}

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
                    if len(examples) == 5:
                        break
                        
            return examples
        else:
            return []
    except Exception as e:
        print(f"Błąd API Tatoeba: {e}")
        return []

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
# INTERFEJS APLIKACJI (GUI)
# ==========================================
async def main():
    st.set_page_config(page_title="Lingilo - Twój pomocnik językowy", layout="wide")
    
    # --- BRAMKA AUTENTYKACJI (AUTH0 OAUTH2) ---
    if "token" not in st.session_state:
        st.title("🔐 Zaloguj się do Lingilo")
        result = oauth2.authorize_button(
            name="Zaloguj przez Auth0",
            icon="https://auth0.com/favicon.ico",
            redirect_uri="http://localhost:8501/component/streamlit_oauth.authorize_button",
            scope="openid email profile"
        )
        
        if result and "token" in result:
            st.session_state.token = result.get("token")
            st.rerun()
            
        st.stop()  
        
    token_data = st.session_state.token
    if "user_info" not in st.session_state:
        try:
            # Używamy standardowej biblioteki requests do odpytania serwera Auth0
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            response = requests.get(f"https://{AUTH0_DOMAIN}/userinfo", headers=headers)
            response.raise_for_status()
            st.session_state.user_info = response.json()
        except Exception as e:
            st.error(f"Błąd komunikacji z Auth0: {e}")
            st.stop()
            
    st.session_state.current_user_id = st.session_state.user_info.get("sub", "default_user")
    current_user = st.session_state.current_user_id
    
    await create_tables()

    # --- ŁADOWANIE PREFERENCJI I ZAPISANYCH DRAFTÓW UŻYTKOWNIKA ---
    if not st.session_state.prefs_loaded:
        prefs = await load_user_preferences(current_user)
        st.session_state.ui_language = prefs.ui_language
        st.session_state.translation_target_language = prefs.translation_target_language
        st.session_state.ai_provider = prefs.ai_provider
        
        if prefs.session_drafts:
            try:
                drafts = json.loads(prefs.session_drafts)
                for k, v in drafts.items():
                    if k in st.session_state:
                        st.session_state[k] = v
                st.session_state.last_saved_drafts = prefs.session_drafts
            except Exception:
                pass
                
        st.session_state.prefs_loaded = True
    
    with st.sidebar:
        st.success("✅ Zalogowano pomyślnie")
        if st.button("Wyloguj"):
            # ZAPISUJEMY DANE SESJI TUŻ PRZED WYLOGOWANIEM
            await save_current_drafts(current_user)
            
            del st.session_state.token
            if "user_info" in st.session_state:
                del st.session_state.user_info
            st.session_state.prefs_loaded = False
            st.rerun()
            
        st.header("⚙️ " + _("Ustawienia"))
        
        ui_langs_translated = [_(l) for l in SUPPORTED_LANGUAGES]
        sel_ui_trans = st.selectbox(
            _("Język interfejsu aplikacji:"), 
            ui_langs_translated, 
            index=SUPPORTED_LANGUAGES.index(st.session_state.ui_language)
        )
        new_ui_lang = SUPPORTED_LANGUAGES[ui_langs_translated.index(sel_ui_trans)]
        
        if new_ui_lang != st.session_state.ui_language:
            st.session_state.ui_language = new_ui_lang
            await update_user_preference(current_user, "ui_language", new_ui_lang)
            st.rerun()
            
        provider_options = ["Groq (Darmowe chmurowe)", "OpenAI (Płatne)"]
        prov_translated = [_(p) for p in provider_options]
        sel_prov_trans = st.selectbox(
            _("Dostawca modelu AI:"), 
            prov_translated, 
            index=provider_options.index(st.session_state.ai_provider)
        )
        new_prov = provider_options[prov_translated.index(sel_prov_trans)]
        if new_prov != st.session_state.ai_provider:
            st.session_state.ai_provider = new_prov
            await update_user_preference(current_user, "ai_provider", new_prov)
            st.rerun()
        
        if st.session_state.ai_provider == "OpenAI (Płatne)":
            st.warning(_("Opcja płatna jest tymczasowo niedostępna. System automatycznie użyje darmowego modelu Groq."))
        
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
            st.session_state.active_task = task
            st.rerun()

    current_task = st.session_state.active_task

    # ----------------------------------------
    # ZAKŁADKA 1: TŁUMACZENIE
    # ----------------------------------------
    if current_task == "Tłumaczenie":
        st.header(_("Wprowadź tekst do przetłumaczenia"))
        
        st.session_state.trans_text = st.text_area(_("Twój tekst:"), value=st.session_state.trans_text)
        
        lang_trans = [_(l) for l in SUPPORTED_LANGUAGES]
        sel_lang_trans = st.selectbox(_("Język docelowy"), lang_trans, index=1)
        lang = SUPPORTED_LANGUAGES[lang_trans.index(sel_lang_trans)]

        if st.button(_("Tłumacz")) and st.session_state.trans_text:
            with st.spinner(_("Trwa tłumaczenie...")):
                st.session_state.translated_text = await monitor_llm(_("Jesteś tłumaczem. Zwróć tylko czyste tłumaczenie."), f"Na {lang}:\n{st.session_state.trans_text}", TaskType.TRANSLATION)
        
        if st.session_state.translated_text:
            st.success(st.session_state.translated_text)
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            
            if c1.button(_("📝 Zapisz jako notatkę")):
                note_content = f"Tłumaczenie ({lang}):\n{st.session_state.trans_text}\n➔\n{st.session_state.translated_text}"
                await save_note_to_db(note_content)
                st.toast(_("Zapisano w notatkach!"), icon="✅")
                
            if c2.button(_("📇 Przekaż do fiszek")):
                st.session_state.flashcard_input_word = st.session_state.trans_text
                st.session_state.active_task = "Powtórki"
                st.rerun()
                
            if c3.button(_("🔊 Przekaż do TTS")):
                st.session_state.tts_text = st.session_state.translated_text
                st.session_state.active_task = "Generowanie audio"
                st.rerun()

    # ----------------------------------------
    # ZAKŁADKA 2: KOREKTA TEKSTU
    # ----------------------------------------
    elif current_task == "Korekta tekstu":
        st.header(_("Korekta błędów i redakcja"))
        st.session_state.corr_text = st.text_area(_("Tekst do poprawy:"), value=st.session_state.corr_text)
        
        if st.button(_("Popraw tekst")) and st.session_state.corr_text:
            with st.spinner(_("Analiza błędów...")):
                sys_prompt = _("Popraw błędy. Zwróć poprawiony tekst.") + " " + _("Następnie w punktach wyjaśnij wprowadzone zmiany i krótko uzasadnij, dlaczego nowa wersja jest lepsza.")
                st.session_state.corrected_text = await monitor_llm(sys_prompt, st.session_state.corr_text, TaskType.CORRECTION)
        
        if st.session_state.corrected_text:
            st.info(st.session_state.corrected_text)
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            if c1.button(_("📝 Zapisz jako notatkę")):
                note_content = f"Korekta:\n{st.session_state.corr_text}\n\nAnaliza:\n{st.session_state.corrected_text}"
                await save_note_to_db(note_content)
                st.toast(_("Zapisano w notatkach!"), icon="✅")
                
            if c2.button(_("🔊 Przekaż do TTS")):
                st.session_state.tts_text = st.session_state.corrected_text
                st.session_state.active_task = "Generowanie audio"
                st.rerun()

    # ----------------------------------------
    # ZAKŁADKA 3: TTS
    # ----------------------------------------
    elif current_task == "Generowanie audio":
        st.header(_("Zamiana tekstu na mowę"))
        st.session_state.tts_text = st.text_area(_("Wpisz tekst:"), value=st.session_state.tts_text)
        
        col1, col2 = st.columns(2)
        if col1.button(_("Generuj wymowę")) and st.session_state.tts_text:
            with st.spinner(_("Przygotowywanie...")):
                try: lang_code = detect(st.session_state.tts_text)
                except: lang_code = "en"
                tts = gTTS(text=st.session_state.tts_text, lang=lang_code)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                st.session_state.audio_generation_audio_file = buf.getvalue()
                
        if st.session_state.audio_generation_audio_file:
            st.audio(st.session_state.audio_generation_audio_file, format="audio/mp3")
            if col2.button(_("Zapisz to Audio do Historii")):
                with st.spinner(_("Zapisywanie i generowanie wektorów...")):
                    emb_json = None
                    if SEMANTIC_SEARCH_AVAILABLE and st.session_state.tts_text:
                        emb = embedder.encode(st.session_state.tts_text).tolist()
                        emb_json = json.dumps(emb)

                    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                        session.add(Note(
                            user_id=current_user,
                            text=f"{_('Wygenerowane TTS')}: {st.session_state.tts_text[:50]}...", 
                            note_type=NoteType.GENERATED_AUDIO, 
                            audio_data=st.session_state.audio_generation_audio_file,
                            embedding=emb_json
                        ))
                        await session.commit()
                st.toast(_("Zapisano audio w historii!"), icon="✅")

    # ----------------------------------------
    # ZAKŁADKA 4: ZAPYTANIE
    # ----------------------------------------
    elif current_task == "Zapytanie":
        st.header(_("Czat z ekspertem językowym"))
        st.session_state.inq_text = st.text_area(_("Zadaj pytanie:"), value=st.session_state.inq_text)
        
        if st.button(_("Zapytaj")) and st.session_state.inq_text:
            with st.spinner(_("Piszę odpowiedź...")):
                st.session_state.inquiry_response = await monitor_llm(_("Jesteś ekspertem językowym."), st.session_state.inq_text, TaskType.INQUIRY)
        
        if st.session_state.inquiry_response:
            st.info(st.session_state.inquiry_response)
            
            if st.button(_("📝 Zapisz jako notatkę")):
                note_content = f"Pytanie: {st.session_state.inq_text}\n\nOdpowiedź:\n{st.session_state.inquiry_response}"
                await save_note_to_db(note_content)
                st.toast(_("Zapisano w notatkach!"), icon="✅")

    # ----------------------------------------
    # ZAKŁADKA 5: NOTATKI
    # ----------------------------------------
    elif current_task == "Notatki":
        st.header(_("Zapisuj mowę i tekst z wyszukiwaniem semantycznym"))
        
        note_audio = st.audio_input(_("🎤 Nagraj notatkę głosową"))
        if note_audio:
            st.session_state.note_audio_bytes = note_audio.getvalue()
            
        c1, c2 = st.columns(2)
        if st.session_state.note_audio_bytes and c1.button(_("Opcjonalnie: Transkrybuj mowę")):
            with st.spinner(_("Rozpoznawanie mowy...")):
                st.session_state.note_audio_text = await transcribe_audio(st.session_state.note_audio_bytes)
                    
        text_note = st.text_area(_("Treść notatki tekstowej (lub transkrypcji):"), value=st.session_state.note_audio_text)
        
        if st.button(_("Zapisz Notatkę do bazy"), type="primary"):
            if not text_note and not st.session_state.note_audio_bytes:
                st.warning(_("Notatka jest pusta!"))
            else:
                with st.spinner(_("Zapisywanie i generowanie wektorów...")):
                    emb_json = None
                    if SEMANTIC_SEARCH_AVAILABLE and text_note:
                        emb = embedder.encode(text_note).tolist()
                        emb_json = json.dumps(emb)
                        
                    n_type = NoteType.AUDIO if st.session_state.note_audio_bytes else NoteType.TEXT
                    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                        session.add(Note(user_id=current_user, text=text_note, note_type=n_type, audio_data=st.session_state.note_audio_bytes, embedding=emb_json))
                        await session.commit()
                st.toast(_("Notatka zapisana pomyślnie!"), icon="✅")
                st.session_state.note_audio_bytes = None
                st.session_state.note_audio_text = ""
                st.rerun()

    # ----------------------------------------
    # ZAKŁADKA 6: POWTÓRKI
    # ----------------------------------------
    elif current_task == "Powtórki":
        st.header(_("Twój system nauki (Spaced Repetition)"))
        
        tab_study, tab_add, tab_edit = st.tabs([_("Nauka (SM-2)"), _("Dodaj Fiszki"), _("Edytuj / Usuń")])
        
        # --- TAB 1: NAUKA ---
        with tab_study:
            async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                now = datetime.now()
                result = await session.execute(
                    select(WordRevision)
                    .where(WordRevision.user_id == current_user)
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
                        
                        if current_card.audio_data:
                            st.audio(current_card.audio_data, format="audio/mp3")

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

        # --- TAB 2: DODAJ FISZKĘ ---
        with tab_add:
            st.subheader(_("Kreator z analizą lingwistyczną AI"))
            
            card_word = st.text_input(_("Wpisz słowo lub wyrażenie:"), value=st.session_state.flashcard_input_word)
            st.session_state.flashcard_input_word = card_word
            
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
            if c1.button(_("Pobierz dane przez AI")) and card_word:
                with st.spinner(_("AI analizuje słowo jako API...")):
                    st.session_state.current_flashcard_draft = await generate_word_data(
                        card_word, card_lang, target_lang
                    )
                    
            if c2.button(_("Pobierz zdania z bazy Tatoeba")) and card_word:
                with st.spinner(_("Szukam autentycznych przykładów...")):
                    api_sents = get_example_sentences_api(
                        card_word, card_lang, target_lang
                    )
                    if api_sents and st.session_state.current_flashcard_draft:
                        st.session_state.current_flashcard_draft['sentences'].extend(api_sents)
                    elif api_sents:
                        st.session_state.current_flashcard_draft = {'sentences': api_sents}
                    else:
                        st.warning(_("API nie znalazło przykładów."))

            if c3.button(_("Wprowadź ręcznie")) and card_word:
                st.session_state.current_flashcard_draft = {
                    'base_form': card_word,
                    'grammar_info': '',
                    'definitions': [],
                    'forms': [],
                    'sentences': []
                }
            
            draft = st.session_state.current_flashcard_draft
            if draft:
                st.markdown("---")
                
                draft_base = st.text_input(_("Forma podstawowa"), value=draft.get('base_form', ''))
                draft['base_form'] = draft_base
                
                draft_grammar = st.text_input(_("Info gramatyczne"), value=draft.get('grammar_info', ''))
                draft['grammar_info'] = draft_grammar
                
                curr_forms = draft.get('forms', [])
                if isinstance(curr_forms, list): curr_forms = ", ".join(curr_forms)
                draft_forms = st.text_input(_("Odmiany"), value=curr_forms)
                draft['forms'] = draft_forms
                
                selected_defs = st.multiselect(_("Wybierz definicje do zapisania:"), draft.get('definitions', []), default=draft.get('definitions', []))
                draft['definitions'] = selected_defs
                
                st.session_state.custom_defs = st.text_area(_("Własne definicje (każda w nowej linii - opcjonalnie):"), placeholder=_("np. definicja 1\ndefinicja 2"), value=st.session_state.custom_defs)
                
                selected_sents = st.multiselect(_("Wybierz przykłady zdań:"), draft.get('sentences', []), default=draft.get('sentences', []))
                draft['sentences'] = selected_sents
                
                st.session_state.custom_sents = st.text_area(_("Własne przykłady (każdy w nowej linii - opcjonalnie):"), placeholder=_("np. To jest mój pierwszy przykład.\nA to jest mój drugi przykład."), value=st.session_state.custom_sents)
                
                st.session_state.custom_notes = st.text_area(_("Twoje własne uwagi (opcjonalnie):"), value=st.session_state.custom_notes)

                st.write(f"**{_('Audio (opcjonalnie):')}**")
                a_col1, a_col2 = st.columns(2)
                with a_col1:
                    if st.button(_("Generuj wymowę (TTS)")):
                        with st.spinner(_("Generowanie...")):
                            try:
                                eng_lang = STATIC_TRANSLATIONS.get("Angielski", {}).get(card_lang, "English")
                                tts_lang_code = langcodes.find(eng_lang).language
                            except:
                                tts_lang_code = "en"
                            tts = gTTS(text=card_word, lang=tts_lang_code)
                            buf = io.BytesIO()
                            tts.write_to_fp(buf)
                            st.session_state.flashcard_audio_bytes = buf.getvalue()
                
                with a_col2:
                    mic_audio = st.audio_input(_("🎤 Nagraj wymowę"))
                    if mic_audio:
                        st.session_state.flashcard_audio_bytes = mic_audio.getvalue()
                
                if st.session_state.flashcard_audio_bytes:
                    st.audio(st.session_state.flashcard_audio_bytes, format="audio/mp3")
                
                if st.button(_("Zapisz gotową fiszkę"), type="primary"):
                    final_defs = selected_defs.copy()
                    for line in st.session_state.custom_defs.split('\n'):
                        if line.strip():
                            final_defs.append(line.strip())
                            
                    final_sents = selected_sents.copy()
                    for line in st.session_state.custom_sents.split('\n'):
                        if line.strip():
                            final_sents.append(line.strip())

                    async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                        new_rev = WordRevision(
                            user_id=current_user,
                            word=card_word, language=card_lang,
                            base_form=draft_base,
                            grammar_info=draft_grammar,
                            definitions=" | ".join(final_defs),
                            inflected_forms=draft_forms,
                            example_sentences=" | ".join(final_sents),
                            custom_notes=st.session_state.custom_notes,
                            audio_data=st.session_state.flashcard_audio_bytes
                        )
                        session.add(new_rev)
                        await session.commit()
                        
                    st.success(_("Fiszka dodana pomyślnie!"))
                    st.session_state.current_flashcard_draft = None
                    st.session_state.flashcard_audio_bytes = None
                    st.session_state.flashcard_input_word = "" 
                    st.session_state.custom_defs = ""
                    st.session_state.custom_sents = ""
                    st.session_state.custom_notes = ""
                    time.sleep(1)
                    st.rerun()

        # --- TAB 3: EDYTUJ / USUŃ ---
        with tab_edit:
            search_query = st.text_input("🔍 " + _("Szukaj w notatkach (po tekście lub znaczeniu):"))
            
            async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                res = await session.execute(
                    select(WordRevision)
                    .where(WordRevision.user_id == current_user)
                    .order_by(WordRevision.creation_date.desc())
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
                
                if not all_cards:
                    st.info(_("Brak historii aktywności."))
                
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
                                c.word = edit_word
                                c.base_form = edit_base
                                c.grammar_info = edit_grammar
                                c.definitions = edit_defs
                                c.inflected_forms = edit_forms
                                c.example_sentences = edit_sents
                                c.custom_notes = edit_notes
                                if delete_audio:
                                    c.audio_data = None
                                await session.commit()
                                st.rerun()
                                
                            if deleted:
                                await session.delete(c)
                                await session.commit()
                                st.rerun()

    # ----------------------------------------
    # ZAKŁADKA 7: HISTORIA I ZAPISANE
    # ----------------------------------------
    elif current_task == "Historia i Zapisane":
        st.header(_("Centrum Danych i Historii"))
        
        t_hist, t_notes = st.tabs([_("Pełna Historia Aktywności"), _("Baza Notatek (Wyszukiwanie Semantyczne)")])
        
        # --- PEŁNA HISTORIA ---
        with t_hist:
            async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                res = await session.execute(
                    select(ActivityLog)
                    .where(ActivityLog.user_id == current_user)
                    .order_by(ActivityLog.created_at.desc())
                    .limit(50)
                )
                logs = res.scalars().all()
                if not logs:
                    st.info(_("Brak historii aktywności."))
                for log in logs:
                    with st.expander(f"{log.created_at.strftime('%Y-%m-%d %H:%M')} | {_(log.task_type.value.upper())}"):
                        st.write(f"**{_('Wejście')}:** {log.input_text}")
                        st.write(f"**{_('Wyjście')}:** {log.output_text}")
        
        # --- NOTATKI Z WYSZUKIWANIEM SEMANTYCZNYM ---
        with t_notes:
            if not SEMANTIC_SEARCH_AVAILABLE:
                st.warning(_("Wyszukiwanie semantyczne jest nieaktywne (wymaga biblioteki sentence-transformers). Działa wyszukiwanie tekstowe."))
                
            q = st.text_input(_("Szukaj w notatkach (po tekście lub znaczeniu):"))
            
            async with sessionmaker(bind=get_engine(), class_=AsyncSession)() as session:
                res = await session.execute(
                    select(Note)
                    .where(Note.user_id == current_user)
                    .order_by(Note.created_at.desc())
                )
                notes = res.scalars().all()
                
                if q and SEMANTIC_SEARCH_AVAILABLE:
                    q_emb = embedder.encode(q).tolist()
                    scored_notes = []
                    for n in notes:
                        if n.embedding:
                            try:
                                n_emb = json.loads(n.embedding)
                                score = cosine_similarity(q_emb, n_emb)
                                scored_notes.append((score, n))
                            except: pass
                    scored_notes.sort(key=lambda x: x[0], reverse=True)
                    notes_to_display = [n for score, n in scored_notes if score > 0.1] 
                elif q:
                    notes_to_display = [n for n in notes if n.text and q.lower() in n.text.lower()]
                else:
                    notes_to_display = notes
                
                for n in notes_to_display:
                    with st.container(border=True):
                        st.caption(f"{n.created_at.strftime('%Y-%m-%d %H:%M')} | {_('Typ')}: {_(n.note_type.value)}")
                        if n.text: st.write(n.text)
                        if n.audio_data: st.audio(n.audio_data, format="audio/wav" if n.note_type == NoteType.AUDIO else "audio/mp3")

    # --- ZAPIS WERSJI ROBOCZYCH (INPUT/OUTPUT) DO BAZY ---
    await save_current_drafts(current_user)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    # Bezpośrednie nałożenie łatki na aktywną pętlę zdarzeń wątku Streamlita
    nest_asyncio.apply(loop)
    
    # Uruchomienie aplikacji za pomocą istniejącej, załatanej pętli
    loop.run_until_complete(main())