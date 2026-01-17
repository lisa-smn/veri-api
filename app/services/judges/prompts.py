"""
Prompt-Templates für LLM-as-a-Judge für verschiedene Dimensionen.

Alle Prompts erzwingen striktes JSON-Output mit festem Schema.
"""


def build_readability_prompt(
    summary_text: str,
    article_text: str | None = None,
    prompt_version: str = "v1",
) -> str:
    """
    Baut Prompt für Readability-Bewertung.

    Input: summary (article optional)
    Output: JSON mit rating (1-5), confidence, rationale
    """
    if prompt_version == "v1":
        return _build_readability_prompt_v1(summary_text, article_text)
    if prompt_version == "v2_float":
        return _build_readability_prompt_v2_float(summary_text, article_text)
    raise ValueError(f"Unbekannte Prompt-Version: {prompt_version}")


def _build_readability_prompt_v1(summary_text: str, article_text: str | None = None) -> str:
    """Readability Prompt v1: Fokus auf Lesbarkeit, 1-5 Rating."""
    article_context = f"\n\n**Artikel (Kontext):**\n{article_text}" if article_text else ""

    return f"""Du bewertest die LESBARKEIT einer Textzusammenfassung.

**Kriterien für Lesbarkeit (Skala 1-5, 5 = exzellent, 1 = sehr schlecht):**
1. **Satzlänge:** Sind die Sätze angemessen lang? (nicht zu lang, nicht zu kurz)
2. **Satzstruktur:** Ist die Struktur klar und nicht übermäßig verschachtelt?
3. **Interpunktion:** Wird Interpunktion angemessen verwendet? (nicht überladen)
4. **Verständlichkeit:** Kann der Text leicht gelesen und verstanden werden?

**WICHTIG:** Nutze die VOLLE Skala (1-5). Nicht nur 4-5 verwenden!

**Zusammenfassung:**
{summary_text}{article_context}

**Aufgabe:**
Bewerte die Lesbarkeit der Zusammenfassung auf einer Skala von 1 bis 5 und gib eine kurze Begründung.

**Output-Schema (MUSS exakt eingehalten werden, NUR JSON):**
```json
{{
  "rating": 3,
  "confidence": 0.8,
  "rationale": "Die Sätze sind angemessen lang, aber einige Verschachtelungen erschweren das Lesen."
}}
```

**WICHTIG:** Antworte NUR mit dem JSON-Objekt. Keine zusätzlichen Kommentare oder Erklärungen außerhalb des JSON."""


def _build_readability_prompt_v2_float(summary_text: str, article_text: str | None = None) -> str:
    """Readability Prompt v2_float: Score 0.00-1.00 mit Ankern und "use full scale"."""
    article_context = f"\n\n**Artikel (Kontext):**\n{article_text}" if article_text else ""

    return f"""Du bewertest die LESBARKEIT einer Textzusammenfassung.

**Kriterien für Lesbarkeit:**
1. **Satzlänge:** Sind die Sätze angemessen lang? (nicht zu lang, nicht zu kurz)
2. **Satzstruktur:** Ist die Struktur klar und nicht übermäßig verschachtelt?
3. **Interpunktion:** Wird Interpunktion angemessen verwendet? (nicht überladen)
4. **Verständlichkeit:** Kann der Text leicht gelesen und verstanden werden?

**Skala: 0.00 bis 1.00 (zwei Dezimalstellen)**

**Anker (Orientierungshilfe):**
- **0.20** = schwer lesbar (holprig, unklar, viele Brüche)
- **0.50** = mittel (verständlich, aber unruhig/uneinheitlich)
- **0.80** = sehr gut (klar, flüssig, gut strukturiert)

**WICHTIG:**
- **Nutze die VOLLE Skala:** Verwende Werte <0.40 und >0.80, wenn passend.
- **Wenn unsicher, verwende nicht automatisch 0.50.**
- **Zwei Dezimalstellen:** z.B. 0.35, 0.67, 0.92

**Zusammenfassung:**
{summary_text}{article_context}

**Aufgabe:**
Bewerte die Lesbarkeit der Zusammenfassung als Score zwischen 0.00 und 1.00 (zwei Dezimalstellen) und gib eine kurze Begründung.

**Output-Schema (MUSS exakt eingehalten werden, NUR JSON):**
```json
{{
  "score": 0.65,
  "confidence": 0.8,
  "rationale": "Die Sätze sind angemessen lang, aber einige Verschachtelungen erschweren das Lesen."
}}
```

**WICHTIG:** Antworte NUR mit dem JSON-Objekt. Keine zusätzlichen Kommentare oder Erklärungen außerhalb des JSON."""


def build_coherence_prompt(
    summary_text: str,
    article_text: str | None = None,
    prompt_version: str = "v1",
) -> str:
    """
    Baut Prompt für Coherence-Bewertung.

    Input: summary (article optional)
    Output: JSON mit rating (1-5), confidence, rationale
    """
    if prompt_version == "v1":
        return _build_coherence_prompt_v1(summary_text, article_text)
    raise ValueError(f"Unbekannte Prompt-Version: {prompt_version}")


def _build_coherence_prompt_v1(summary_text: str, article_text: str | None = None) -> str:
    """Coherence Prompt v1: Fokus auf logischen Fluss, keine Widersprüche."""
    article_context = f"\n\n**Artikel (Kontext):**\n{article_text}" if article_text else ""

    return f"""Du bewertest die KOHÄRENZ einer Textzusammenfassung.

**Kriterien für Kohärenz (Skala 1-5, 5 = exzellent, 1 = sehr schlecht):**
1. **Logischer Fluss:** Sind die Sätze und Absätze logisch miteinander verbunden?
2. **Keine Widersprüche:** Enthält die Zusammenfassung keine widersprüchlichen Informationen?
3. **Referenzklarheit:** Sind Pronomen und andere Verweise eindeutig?
4. **Struktur:** Gibt es keine abrupten Sprünge in den Themen?

**WICHTIG:** Nutze die VOLLE Skala (1-5). Nicht nur 4-5 verwenden!

**Zusammenfassung:**
{summary_text}{article_context}

**Aufgabe:**
Bewerte die Kohärenz der Zusammenfassung auf einer Skala von 1 bis 5 und gib eine kurze Begründung.

**Output-Schema (MUSS exakt eingehalten werden, NUR JSON):**
```json
{{
  "rating": 3,
  "confidence": 0.8,
  "rationale": "Die Zusammenfassung hat einen grundsätzlich logischen Fluss, aber einige Übergänge fehlen."
}}
```

**WICHTIG:** Antworte NUR mit dem JSON-Objekt. Keine zusätzlichen Kommentare oder Erklärungen außerhalb des JSON."""


def build_factuality_prompt(
    summary_text: str,
    article_text: str,
    prompt_version: str = "v1",
) -> str:
    """
    Baut Prompt für Factuality-Bewertung.

    Input: article + summary (beide erforderlich)
    Output: JSON mit rating (1-5) oder error_present (binary), confidence, rationale
    """
    if prompt_version == "v1":
        return _build_factuality_prompt_v1(summary_text, article_text)
    if prompt_version == "v2_binary":
        return _build_factuality_prompt_v2_binary(summary_text, article_text)
    raise ValueError(f"Unbekannte Prompt-Version: {prompt_version}")


def _build_factuality_prompt_v1(summary_text: str, article_text: str) -> str:
    """Factuality Prompt v1: Bewertet Faktentreue anhand des Artikels."""
    return f"""Du bewertest die FAKTENTREUE einer Textzusammenfassung anhand des bereitgestellten Artikels.

**Kriterien für Faktentreue (Skala 1-5, 5 = exzellent, 1 = sehr schlecht):**
1. **Korrekte Fakten:** Sind alle Fakten im Artikel korrekt wiedergegeben?
2. **Keine Erfindungen:** Werden keine Fakten hinzugefügt, die nicht im Artikel stehen?
3. **Korrekte Entitäten:** Sind Namen, Orte, Zahlen korrekt?
4. **Keine Verzerrungen:** Wird der Inhalt nicht verzerrt oder missverständlich dargestellt?

**WICHTIG:** Nutze die VOLLE Skala (1-5). Nicht nur 4-5 verwenden!

**Artikel:**
{article_text}

**Zusammenfassung:**
{summary_text}

**Aufgabe:**
Bewerte die Faktentreue der Zusammenfassung anhand des Artikels auf einer Skala von 1 bis 5 und gib eine kurze Begründung.

**Output-Schema (MUSS exakt eingehalten werden, NUR JSON):**
```json
{{
  "rating": 3,
  "confidence": 0.8,
  "rationale": "Die meisten Fakten sind korrekt, aber eine Zahl wurde falsch wiedergegeben."
}}
```

**WICHTIG:** Antworte NUR mit dem JSON-Objekt. Keine zusätzlichen Kommentare oder Erklärungen außerhalb des JSON."""


def _build_factuality_prompt_v2_binary(summary_text: str, article_text: str) -> str:
    """Factuality Prompt v2_binary: Direktes binary verdict (error_present: true/false)."""
    return f"""Du bewertest die FAKTENTREUE einer Textzusammenfassung anhand des bereitgestellten Artikels.

**Aufgabe:**
Prüfe, ob die Zusammenfassung FAKTISCHE FEHLER enthält (Informationen, die nicht im Artikel stehen oder falsch wiedergegeben sind).

**Kriterien für Fehler:**
- **Falsche Fakten:** Zahlen, Namen, Orte, Daten, die im Artikel nicht vorkommen oder falsch sind
- **Erfundene Informationen:** Behauptungen, die nicht im Artikel stehen
- **Verzerrungen:** Informationen, die den Artikelinhalt verzerren oder missverständlich darstellen
- **NICHT als Fehler zählen:** Paraphrasierungen, Zusammenfassungen, stilistische Unterschiede (solange der Inhalt korrekt ist)

**WICHTIG:**
- Setze `error_present=true` NUR wenn du einen KLAREN faktischen Fehler identifizierst
- Wenn unsicher, setze `error_present=false` und `confidence` niedrig
- `confidence` sollte hoch sein, wenn du dir sicher bist, niedrig bei Unsicherheit

**Artikel:**
{article_text}

**Zusammenfassung:**
{summary_text}

**Aufgabe:**
Entscheide, ob die Zusammenfassung faktische Fehler enthält, und gib eine kurze Begründung.

**Output-Schema (MUSS exakt eingehalten werden, STRICT JSON, keine Prosa):**
```json
{{
  "error_present": false,
  "confidence": 0.9,
  "rationale": "Alle Fakten sind korrekt wiedergegeben, keine Erfindungen oder Verzerrungen."
}}
```

**WICHTIG:**
- Antworte NUR mit dem JSON-Objekt. Keine zusätzlichen Kommentare oder Erklärungen außerhalb des JSON.
- `error_present`: boolean (true/false)
- `confidence`: float zwischen 0.0 und 1.0 (zwei Dezimalstellen empfohlen)
- `rationale`: string (optional, kann leer sein)"""
