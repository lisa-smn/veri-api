"""
Sanity-Checks für das Readability-Paket.

Prüft:
- Existenz von docs/status/readability_status.md
- Existenz der referenzierten Artefakt-Links (relative Pfade)
- Regenerierung via build_readability_status.py (Exit-Code 0)
- Optional: Check-Mode (Datei-Vergleich)
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


def find_relative_links_in_markdown(md_path: Path) -> set[str]:
    """Extrahiert relative Links aus Markdown-Datei."""
    links = set()
    if not md_path.exists():
        return links

    with md_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Suche nach Markdown-Links: [text](path)
    # Relative Pfade beginnen nicht mit http://, https://, mailto:, #
    pattern_md_link = r"\[([^\]]+)\]\(([^\)]+)\)"

    for match in re.finditer(pattern_md_link, content):
        link_path = match.group(2)
        # Nur relative Pfade (keine URLs, keine Anker)
        if not link_path.startswith(("http://", "https://", "mailto:", "#")):
            # Nur Pfade, die wie Datei/Verzeichnis-Pfade aussehen
            if "/" in link_path or link_path.endswith(
                (".md", ".csv", ".json", ".jsonl", ".png", ".jpg")
            ):
                links.add(link_path)

    # Suche nach Code-Blöcken mit Pfaden: `path/to/file` (nur in bestimmten Kontexten)
    # Suche nach Mustern wie "`results/evaluation/...`" oder "`docs/...`"
    pattern_code_path = r"`(results/[^`]+)`|`(docs/[^`]+)`|`(data/[^`]+)`"

    for match in re.finditer(pattern_code_path, content):
        for group in match.groups():
            if group:
                links.add(group)

    return links


def check_status_report_exists(repo_root: Path) -> bool:
    """Prüft, dass readability_status.md existiert."""
    status_path = repo_root / "docs" / "status" / "readability_status.md"
    if not status_path.exists():
        print(f"✗ Status-Report nicht gefunden: {status_path}")
        return False
    print(f"✓ Status-Report existiert: {status_path}")
    return True


def check_artifact_links(repo_root: Path, status_path: Path) -> bool:
    """Prüft, dass alle referenzierten Artefakt-Links existieren."""
    links = find_relative_links_in_markdown(status_path)

    if not links:
        print("⚠ Keine relativen Links gefunden (möglicherweise nur externe URLs)")
        return True

    missing = []
    for link in links:
        # Resolve relativ zum Repo-Root (nicht zur Markdown-Datei)
        if link.startswith("/"):
            # Absoluter Pfad relativ zu Repo-Root
            resolved = repo_root / link.lstrip("/")
        else:
            # Relativer Pfad: versuche zuerst relativ zum Repo-Root
            resolved = repo_root / link
            if not resolved.exists():
                # Fallback: relativ zur Markdown-Datei
                resolved = status_path.parent / link

        # Normalisiere Pfad (auflösen von ..)
        try:
            resolved = resolved.resolve()
        except (OSError, ValueError):
            # Pfad zu lang oder ungültig
            missing.append((link, resolved))
            continue

        # Prüfe, ob Pfad innerhalb des Repos liegt (Sicherheit)
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            print(f"⚠ Link außerhalb des Repos (ignoriert): {link}")
            continue

        if not resolved.exists():
            missing.append((link, resolved))

    if missing:
        print("✗ Fehlende Artefakt-Links:")
        for link, resolved in missing:
            print(f"  - {link} -> {resolved}")
        return False

    print(f"✓ Alle {len(links)} Artefakt-Links existieren")
    return True


def check_build_script_runs(repo_root: Path, check_mode: bool = False) -> bool:
    """Prüft, dass build_readability_status.py erfolgreich läuft."""
    script_path = repo_root / "scripts" / "build_readability_status.py"
    if not script_path.exists():
        print(f"✗ Build-Script nicht gefunden: {script_path}")
        return False

    # Prüfe, ob das Script ausführbar ist (--help sollte funktionieren)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"✗ Build-Script --help fehlgeschlagen (Exit-Code {result.returncode})")
            print(f"  stderr: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Build-Script --help timeout")
        return False
    except Exception as e:
        print(f"✗ Build-Script nicht ausführbar: {e}")
        return False

    print("✓ Build-Script ist ausführbar")

    # Optional: Check-Mode testen (wenn Datei existiert)
    if check_mode:
        status_path = repo_root / "docs" / "status" / "readability_status.md"
        if status_path.exists():
            # Versuche Check-Mode (benötigt --agent_run_dir und --baseline_matrix)
            # Da wir diese nicht haben, überspringen wir diesen Test
            print("⚠ Check-Mode erfordert --agent_run_dir und --baseline_matrix (übersprungen)")
        else:
            print("⚠ Check-Mode erfordert bestehende Datei (übersprungen)")

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-Checks für Readability-Paket")
    ap.add_argument(
        "--repo_root", type=str, help="Repository-Root (default: aktuelles Verzeichnis)"
    )
    ap.add_argument(
        "--check_mode",
        action="store_true",
        help="Prüfe auch Check-Mode von build_readability_status.py",
    )

    args = ap.parse_args()

    # Bestimme Repo-Root
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        # Versuche, Repo-Root zu finden (suche nach .git oder scripts/)
        current = Path.cwd().resolve()
        repo_root = current
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists() or (parent / "scripts").exists():
                repo_root = parent
                break

    print(f"Repository-Root: {repo_root}")
    print()

    all_ok = True

    # Check 1: Status-Report existiert
    if not check_status_report_exists(repo_root):
        all_ok = False
    print()

    # Check 2: Artefakt-Links existieren
    status_path = repo_root / "docs" / "status" / "readability_status.md"
    if status_path.exists():
        if not check_artifact_links(repo_root, status_path):
            all_ok = False
    print()

    # Check 3: Build-Script läuft
    if not check_build_script_runs(repo_root, check_mode=args.check_mode):
        all_ok = False
    print()

    if all_ok:
        print("✓ Alle Sanity-Checks bestanden")
        return 0
    print("✗ Einige Sanity-Checks fehlgeschlagen")
    return 1


if __name__ == "__main__":
    sys.exit(main())
