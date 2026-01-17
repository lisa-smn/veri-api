#!/bin/bash
# Single entry point für alle Readability-Checks
# Führt alle Sanity-Checks und Tests aus

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "🔍 Readability-Paket: Vollständige Checks"
echo "=========================================="
echo ""

# Schritt 1: pytest
echo "1️⃣  Führe pytest-Tests aus..."
python -m pytest -q tests/readability/ || {
    echo "❌ pytest-Tests fehlgeschlagen"
    exit 1
}
echo "✅ pytest-Tests bestanden"
echo ""

# Schritt 2: Sanity-Checks
echo "2️⃣  Führe Sanity-Checks aus..."
python scripts/check_readability_package.py || {
    echo "❌ Sanity-Checks fehlgeschlagen"
    exit 1
}
echo "✅ Sanity-Checks bestanden"
echo ""

# Schritt 3: Status-Report Check (mit Mini-Fixtures für Determinismus)
if [ -d "tests/fixtures/readability_run_mini" ] && [ -f "tests/fixtures/readability_baseline_matrix_mini.csv" ]; then
    echo "3️⃣  Prüfe Status-Report-Konsistenz (mit Mini-Fixtures)..."
    if [ -f "tests/fixtures/readability_status_expected.md" ]; then
        python scripts/build_readability_status.py \
            --agent_run_dir tests/fixtures/readability_run_mini \
            --baseline_matrix tests/fixtures/readability_baseline_matrix_mini.csv \
            --out /tmp/readability_status_generated.md \
            --check tests/fixtures/readability_status_expected.md || {
            echo "❌ Status-Report-Check fehlgeschlagen"
            exit 1
        }
        echo "✅ Status-Report-Check bestanden"
    else
        echo "⚠️  Erwartete Datei nicht gefunden (tests/fixtures/readability_status_expected.md), überspringe Check"
    fi
else
    echo "⚠️  Mini-Fixtures nicht verfügbar, überspringe Status-Report-Check"
fi
echo ""

echo "✅ Alle Checks bestanden!"
echo ""

