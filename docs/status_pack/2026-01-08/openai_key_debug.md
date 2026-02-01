# OpenAI API Key Debug Report

**Datum:** 2026-01-30  
**Zweck:** Diagnose und Fix für 401 Invalid API Key Fehler

---

## Problem

**Symptom:**
- Coherence Evaluation schlägt fehl mit 401 Invalid API Key
- Fehlermeldung zeigt Platzhalter: `"your-ope************here"`
- Alle 1700 Beispiele schlagen fehl (n_used=0, n_failed=1700)

**Fehlerstelle:**
- `app/llm/openai_client.py:15` - `OpenAI()` wird ohne expliziten API-Key aufgerufen
- Client liest automatisch aus `OPENAI_API_KEY` Environment Variable
- Wenn Key fehlt oder Platzhalter ist, wird dieser an OpenAI gesendet → 401 Error

---

## Root Cause Analysis

### 1. API-Key-Laden

**Aktueller Code (vor Fix):**
```python
class OpenAIClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Liest OPENAI_API_KEY automatisch aus der Umgebung
        self.client = OpenAI()  # ← Keine Validierung!
```

**Problem:**
- `OpenAI()` liest automatisch aus `os.getenv("OPENAI_API_KEY")`
- Keine Validierung ob Key vorhanden/valide ist
- Platzhalter-Key wird direkt an API gesendet → 401 Error

### 2. Platzhalter-Quellen

**Gefundene Platzhalter-Strings:**
- `docs/status/readability_status.md:115`: `"your-key-here"`
- `.env.example`: Möglicherweise Platzhalter (nicht lesbar in Sandbox)

**Mögliche Ursachen:**
1. `.env` Datei enthält Platzhalter aus `.env.example`
2. `.env` Datei existiert nicht → `load_dotenv()` findet nichts
3. Environment Variable nicht gesetzt → `os.getenv()` gibt `None`
4. OpenAI Client verwendet `None` oder Platzhalter als Default

---

## Fix Implementation

### 1. Debug-Logging hinzugefügt

**Datei:** `app/llm/openai_client.py:12-60`

**Funktionen:**
- `mask_api_key()`: Maskiert Key für sicheres Logging (erste 6 + letzte 4 Zeichen)
- `validate_api_key()`: Validiert Key und wirft Exception bei Problemen

**Logging:**
```python
print(f"[OpenAIClient] OPENAI_API_KEY(masked)={mask_api_key(api_key)} | source={key_source} | process=script")
```

### 2. Fail-Fast Schutz

**Validierungen:**
1. **Key vorhanden:** `if not key: raise ValueError(...)`
2. **Platzhalter-Patterns:** Prüft auf `"your-ope"`, `"your-openai"`, `"your-key"`, `"sk-proj-"`, `"********"`, `"changeme"`, etc.
3. **Minimale Länge:** `if len(key) < 20: raise ValueError(...)`
4. **Gültiges Prefix:** `if not key.startswith("sk-"): raise ValueError(...)`

**Vorteile:**
- Fehler wird sofort beim Client-Init erkannt (nicht erst bei API-Call)
- Klare Fehlermeldung mit maskiertem Key
- Verhindert, dass Platzhalter an OpenAI gesendet werden

### 3. Expliziter API-Key

**Vorher:**
```python
self.client = OpenAI()  # Liest automatisch aus ENV
```

**Nachher:**
```python
validated_key = validate_api_key(api_key)
self.client = OpenAI(api_key=validated_key)  # Explizit übergeben
```

**Vorteil:** Bessere Kontrolle und Fehlerbehandlung

---

## Code Changes

### `app/llm/openai_client.py`

**Hinzugefügt:**
- `mask_api_key()` Funktion (Zeilen 12-20)
- `validate_api_key()` Funktion (Zeilen 23-66)
- Debug-Logging in `__init__` (Zeilen 75-84)
- Fail-fast Validierung (Zeile 87)
- Expliziter API-Key (Zeile 90)

**Geändert:**
- `__init__` Methode: Validiert Key vor Client-Erstellung

---

## Verifizierung

### Test 1: Fehlender Key

**Erwartet:** `ValueError: OPENAI_API_KEY is not set...`

```python
# Temporär Key entfernen
os.environ.pop("OPENAI_API_KEY", None)
client = OpenAIClient(model_name="gpt-4o-mini")
# → ValueError mit klarer Meldung
```

### Test 2: Platzhalter-Key

**Erwartet:** `ValueError: OPENAI_API_KEY appears to be a placeholder...`

```python
os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
client = OpenAIClient(model_name="gpt-4o-mini")
# → ValueError mit maskiertem Key
```

### Test 3: Gültiger Key

**Erwartet:** Client wird erfolgreich erstellt, Log zeigt maskierten Key

```python
os.environ["OPENAI_API_KEY"] = "sk-valid-key-here..."
client = OpenAIClient(model_name="gpt-4o-mini")
# → [OpenAIClient] OPENAI_API_KEY(masked)=sk-val...here | source=ENV | process=script
# → Client erstellt erfolgreich
```

---

## Log-Auszug (Beispiel)

**Bei erfolgreichem Key:**
```
[OpenAIClient] OPENAI_API_KEY(masked)=sk-proj…xyz1 | source=ENV | process=script
```

**Bei fehlendem Key:**
```
[OpenAIClient] OPENAI_API_KEY(masked)=<MISSING> | source=.env (but key not found) | process=script
ValueError: OPENAI_API_KEY is not set. Set it in .env file or as environment variable.
```

**Bei Platzhalter-Key:**
```
[OpenAIClient] OPENAI_API_KEY(masked)=your-o…here | source=ENV | process=script
ValueError: OPENAI_API_KEY appears to be a placeholder: 'your-o…here'. Please set a valid API key.
```

---

## Nächste Schritte

1. ✅ **Debug-Logging implementiert** - Zeigt maskierten Key und Quelle
2. ✅ **Fail-fast Schutz implementiert** - Validiert Key vor Client-Erstellung
3. ⏳ **Test mit gültigem Key** - Nach Setzen von `OPENAI_API_KEY` in `.env`
4. ⏳ **Coherence Evaluation re-run** - Sollte jetzt erfolgreich sein

---

## Zusammenfassung

**Ursache:**
- `OpenAI()` Client wurde ohne Validierung erstellt
- Platzhalter-Key oder fehlender Key wurde an API gesendet → 401 Error

**Fix:**
- Debug-Logging mit maskiertem Key
- Fail-fast Validierung (Key vorhanden, kein Platzhalter, gültiges Format)
- Expliziter API-Key-Übergabe an Client

**Datei:** `app/llm/openai_client.py`  
**Zeilen:** 12-90 (neue Funktionen + geänderte `__init__`)
