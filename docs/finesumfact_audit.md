# FineSumFact Workspace Audit

**Datum:** 2025-01-07  
**Zweck:** Analyse der Beziehung zwischen FineSumFact und veri-api Workspaces

---

## 1. Projekt-Pfade

### veri-api
- **Absoluter Pfad:** `/Users/lisasimon/PycharmProjects/veri-api`
- **Git-Repo:** ✅ Ja
- **Remote:** `https://github.com/lisa-smn/veri-api.git`
- **Letzter Commit:** `07457c3` - "tests: align factuality API + meta-sentence tests with current agent output"

### FineSumFact
- **Absoluter Pfad:** `/Users/lisasimon/PycharmProjects/FineSumFact`
- **Git-Repo:** ✅ Ja
- **Remote:** `https://github.com/DISL-Lab/FineSumFact.git`
- **Letzter Commit:** `4feae59` - "Update README.md"

---

## 2. Git-Status

### veri-api
- ✅ **Git-Repo:** Aktiv
- ✅ **Remote konfiguriert:** `origin` → `https://github.com/lisa-smn/veri-api.git`
- ✅ **Kein nested .git:** Nur ein `.git`-Verzeichnis im Root (`/Users/lisasimon/PycharmProjects/veri-api/.git`)
- ✅ **Kein Submodule:** FineSumFact ist **nicht** als Git-Submodule eingebunden

### FineSumFact
- ✅ **Git-Repo:** Aktiv (separates Repository)
- ✅ **Remote konfiguriert:** `origin` → `https://github.com/DISL-Lab/FineSumFact.git`
- ⚠️ **Separates Repo:** FineSumFact ist ein **eigenständiges Git-Repository** (nicht Teil von veri-api)

---

## 3. FineSumFact-Bezug in veri-api

### Dateien und Verzeichnisse

**Verzeichnis:**
- `veri-api/data/finesumfact/` existiert
- Enthält konvertierte/verarbeitete Daten:
  - `human_label_test.json` (3.3 MB)
  - `human_label_test_clean.jsonl` (3.3 MB) - **Konvertiert**
  - `human_label_train.json` (19.6 MB)
  - `human_label_train_clean.jsonl` (20.1 MB) - **Konvertiert**

**Scripts:**
- `scripts/convert_finesumfact.py` - Konvertiert FineSumFact-Daten (JSON → JSONL)

**Konfiguration:**
- `configs/m10_factuality_runs.yaml` - Verwendet `data/finesumfact/human_label_test_clean.jsonl`
- Viele Dokumentations-Dateien referenzieren FineSumFact

**Git-Status:**
- `data/finesumfact/` ist **nicht** in `.gitignore`
- Dateien sind **nicht** im Git-Index (keine uncommitted changes)
- ⚠️ **Große Dateien:** ~47 MB JSON/JSONL-Dateien (potentiell problematisch für Git)

---

## 4. Vergleich: FineSumFact Original vs veri-api

### FineSumFact Original (`/Users/lisasimon/PycharmProjects/FineSumFact/dataset/`)
- `human_label_test.json` (3.3 MB) - ✅ **Gleich**
- `human_label_train.json` (19.6 MB) - ✅ **Gleich**
- `machine_label_train.json` (640.9 MB) - ❌ **Nur in Original**

### veri-api (`/Users/lisasimon/PycharmProjects/veri-api/data/finesumfact/`)
- `human_label_test.json` (3.3 MB) - ✅ **Kopie**
- `human_label_test_clean.jsonl` (3.3 MB) - ✅ **Konvertiert (nur in veri-api)**
- `human_label_train.json` (19.6 MB) - ✅ **Kopie**
- `human_label_train_clean.jsonl` (20.1 MB) - ✅ **Konvertiert (nur in veri-api)**

**Fazit:**
- veri-api hat **Kopien** der Original-Daten
- veri-api hat **zusätzliche konvertierte JSONL-Dateien** (für Evaluation verwendet)
- FineSumFact Original enthält **zusätzlich** `machine_label_train.json` (640 MB, nicht in veri-api)

---

## 5. Ergebnis

### Status: **Duplikat (Kopie, kein Submodule)**

**Situation:**
- FineSumFact und veri-api sind **zwei separate Git-Repositories**
- veri-api enthält **Kopien** von FineSumFact-Daten in `data/finesumfact/`
- **Kein Git-Submodule:** FineSumFact ist nicht als Submodule eingebunden
- **Kein nested .git:** Kein `.git`-Verzeichnis innerhalb von `veri-api/data/finesumfact/`

**Probleme:**
1. ⚠️ **Duplikation:** Daten existieren in beiden Repos (Redundanz)
2. ⚠️ **Synchronisation:** Änderungen in FineSumFact müssen manuell nach veri-api kopiert werden
3. ⚠️ **Große Dateien:** ~47 MB JSON/JSONL in veri-api (potentiell problematisch für Git)
4. ⚠️ **Versionierung:** Keine automatische Versionskontrolle der FineSumFact-Daten in veri-api

---

## 6. Empfehlungen

### Option 1: **Ordner-Merge (Empfohlen für aktuelle Situation)**

**Beschreibung:** FineSumFact als einfachen Ordner in veri-api behalten (aktueller Zustand).

**Vorteile:**
- ✅ Einfach (keine Git-Konfiguration nötig)
- ✅ Funktioniert bereits
- ✅ Keine Abhängigkeiten zu externem Repo

**Nachteile:**
- ❌ Duplikation der Daten
- ❌ Manuelle Synchronisation nötig
- ❌ Große Dateien im Git-Repo

**Wann geeignet:**
- Wenn FineSumFact-Daten **selten** aktualisiert werden
- Wenn **nur konvertierte JSONL-Dateien** benötigt werden (nicht Original-JSON)
- Wenn **keine automatische Synchronisation** erforderlich ist

**Schritte:** Keine Änderung nötig (aktueller Zustand)

---

### Option 2: **Git Submodule (Empfohlen für langfristige Wartung)**

**Beschreibung:** FineSumFact als Git-Submodule in veri-api einbinden.

**Vorteile:**
- ✅ Automatische Versionskontrolle (Commit-Hash)
- ✅ Einfaches Update: `git submodule update --remote`
- ✅ Keine Duplikation (nur Referenz)
- ✅ Klare Abhängigkeit dokumentiert

**Nachteile:**
- ⚠️ Zusätzliche Git-Konfiguration
- ⚠️ Team-Mitglieder müssen Submodule initialisieren (`git submodule update --init`)
- ⚠️ Konvertierte JSONL-Dateien müssen weiterhin in veri-api bleiben (oder Build-Step)

**Wann geeignet:**
- Wenn FineSumFact-Daten **regelmäßig** aktualisiert werden
- Wenn **Team-Zusammenarbeit** wichtig ist
- Wenn **Versionierung** der Original-Daten wichtig ist

**Schritte:**
```bash
# 1. Backup der aktuellen Daten
cd /Users/lisasimon/PycharmProjects/veri-api
cp -r data/finesumfact data/finesumfact_backup

# 2. Entferne alte Kopie (nicht löschen, nur aus Git entfernen)
git rm -r --cached data/finesumfact

# 3. Füge FineSumFact als Submodule hinzu
git submodule add https://github.com/DISL-Lab/FineSumFact.git data/finesumfact_original

# 4. Konvertierte Dateien bleiben in data/finesumfact/
# (oder umbenennen: data/finesumfact_original → data/finesumfact, dann Submodule-Pfad anpassen)

# 5. .gitmodules prüfen und committen
git add .gitmodules
git commit -m "Add FineSumFact as submodule"
```

**Hinweis:** Konvertierte JSONL-Dateien sollten weiterhin in veri-api bleiben (Build-Artefakt), Original-JSON aus Submodule lesen.

---

### Option 3: **Subtree (Alternative zu Submodule)**

**Beschreibung:** FineSumFact-Daten direkt in veri-api committen (ohne Submodule-Overhead).

**Vorteile:**
- ✅ Einfacher als Submodule (keine Initialisierung nötig)
- ✅ Alle Daten in einem Repo
- ✅ Keine Duplikation (nur ein Ort)

**Nachteile:**
- ❌ Git-Historie wird vermischt
- ❌ Updates sind komplexer (`git subtree pull`)
- ❌ Große Dateien bleiben im Repo

**Wann geeignet:**
- Wenn FineSumFact-Daten **selten** aktualisiert werden
- Wenn **einfache Struktur** wichtiger ist als saubere Git-Historie

**Schritte:**
```bash
# 1. Backup
cd /Users/lisasimon/PycharmProjects/veri-api
cp -r data/finesumfact data/finesumfact_backup

# 2. Entferne alte Kopie
git rm -r --cached data/finesumfact

# 3. Füge FineSumFact als Subtree hinzu
git subtree add --prefix=data/finesumfact_original \
    https://github.com/DISL-Lab/FineSumFact.git main --squash

# 4. Committen
git commit -m "Add FineSumFact as subtree"
```

---

## 7. Konkrete Schritte (ohne Datenverlust)

### Empfehlung: **Option 1 (Ordner-Merge) beibehalten**

**Begründung:**
- Aktuelle Struktur funktioniert
- Konvertierte JSONL-Dateien sind Build-Artefakte (sollten in veri-api bleiben)
- FineSumFact-Daten werden selten aktualisiert
- Keine komplexe Git-Konfiguration nötig

**Optimierungen (optional):**

1. **Große Dateien aus Git entfernen (Git LFS oder .gitignore):**
   ```bash
   cd /Users/lisasimon/PycharmProjects/veri-api
   
   # Füge zu .gitignore hinzu:
   echo "data/finesumfact/*.json" >> .gitignore
   echo "data/finesumfact/*.jsonl" >> .gitignore
   echo "!data/finesumfact/.gitkeep" >> .gitignore
   
   # Entferne große Dateien aus Git (falls bereits committed)
   git rm --cached data/finesumfact/*.json data/finesumfact/*.jsonl
   git commit -m "Remove large FineSumFact files from Git (use .gitignore)"
   ```

2. **Dokumentation aktualisieren:**
   - In README.md dokumentieren: "FineSumFact-Daten müssen manuell aus `/Users/lisasimon/PycharmProjects/FineSumFact` kopiert werden"
   - Konvertierungs-Script dokumentieren: `scripts/convert_finesumfact.py`

3. **Automatisierung (optional):**
   ```bash
   # Script: scripts/sync_finesumfact.sh
   #!/bin/bash
   SOURCE="/Users/lisasimon/PycharmProjects/FineSumFact/dataset"
   TARGET="/Users/lisasimon/PycharmProjects/veri-api/data/finesumfact"
   
   # Kopiere Original-Daten
   cp "$SOURCE/human_label_test.json" "$TARGET/"
   cp "$SOURCE/human_label_train.json" "$TARGET/"
   
   # Konvertiere zu JSONL
   python3 scripts/convert_finesumfact.py
   ```

---

## 8. Zusammenfassung

| Aspekt | Status | Empfehlung |
|--------|--------|------------|
| **Git-Status** | Zwei separate Repos | ✅ OK (aktueller Zustand) |
| **Duplikation** | Daten in beiden Repos | ⚠️ Akzeptabel (seltene Updates) |
| **Nested .git** | Kein nested .git | ✅ OK (kein Submodule) |
| **Große Dateien** | ~47 MB in veri-api | ⚠️ Optional: .gitignore oder Git LFS |
| **Synchronisation** | Manuell | ⚠️ Optional: Sync-Script |

**Fazit:**
- ✅ **Aktuelle Struktur ist funktional** (keine kritischen Probleme)
- ⚠️ **Optimierungen möglich** (Git LFS, .gitignore, Sync-Script)
- ✅ **Keine sofortigen Änderungen erforderlich**

**Empfehlung:** Beibehalten der aktuellen Struktur (Option 1), mit optionalen Optimierungen für große Dateien.

