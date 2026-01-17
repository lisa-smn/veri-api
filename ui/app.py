"""Streamlit Dashboard für veri-api."""

import os
from pathlib import Path
import sys

import streamlit as st

# Add ui directory to path
UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR))

import api_client
import dataset_loader
import db_client
import docs_view
import injector
import render


# Caching für API-Aufrufe
@st.cache_data(ttl=300)  # 5 Minuten Cache
def get_cached_datasets():
    """Gecachte Dataset-Liste."""
    try:
        return api_client.get_datasets()
    except Exception:
        return api_client.DEFAULT_DATASETS.copy()


@st.cache_data(ttl=300)  # 5 Minuten Cache
def get_cached_models():
    """Gecachte Model-Liste."""
    try:
        return api_client.get_models()
    except Exception:
        return api_client.DEFAULT_MODELS.copy()


# Page config
st.set_page_config(
    page_title="veri-api Dashboard",
    page_icon="🔍",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("🔍 veri-api Dashboard")

    # API Health
    st.subheader("API Status")
    api_health = api_client.health_check()
    if api_health.get("available"):
        st.success(f"✅ API: {api_client.API_BASE_URL}")
    else:
        st.error(f"❌ API: {api_client.API_BASE_URL}")
        st.caption(f"Fehler: {api_health.get('message', 'Unknown')}")

    # DB Status
    st.subheader("Database Status")
    pg_available = db_client.is_available()
    if pg_available:
        st.success("✅ Postgres")
    else:
        st.warning("⚠️ Postgres nicht verfügbar")
        st.caption("Prüfe POSTGRES_DSN oder einzelne ENV-Vars")

    # Neo4j Status (optional)
    neo4j_uri = os.getenv("NEO4J_URI")
    if neo4j_uri:
        st.info(f"Neo4j: {neo4j_uri}")
    else:
        st.caption("Neo4j: nicht konfiguriert")

    st.divider()

    # Debug Toggle
    show_debug = st.checkbox(
        "Show debug", value=False, key="show_debug", help="Zeige Debug-Informationen an"
    )

    st.divider()
    st.caption(f"API Base URL: {api_client.API_BASE_URL}")


# Main Tabs
tab1, tab2, tab3 = st.tabs(["Verify", "Runs", "Status"])

# Tab 1: Verify
with tab1:
    st.header("Verification")
    st.caption("Sende Article + Summary an die API zur Verifikation")

    # Initialisiere Session State für Textfelder (vor dem Rendern!)
    if "article_text" not in st.session_state:
        st.session_state["article_text"] = ""
    if "summary_text" not in st.session_state:
        st.session_state["summary_text"] = ""
    if "loaded_example_meta" not in st.session_state:
        st.session_state["loaded_example_meta"] = None

    # Pending updates für Textareas (verhindert StreamlitAPIException)
    if "pending_article_text" not in st.session_state:
        st.session_state["pending_article_text"] = None
    if "pending_summary_text" not in st.session_state:
        st.session_state["pending_summary_text"] = None

    # Advanced Options (collapsed) - ZUERST, damit Button-Click verarbeitet werden kann
    with st.expander("Advanced Options"):
        # Dataset Dropdown (mit Caching)
        try:
            datasets = get_cached_datasets()
            use_fallback_datasets = False
        except Exception:
            datasets = api_client.DEFAULT_DATASETS.copy()
            use_fallback_datasets = True

        # Füge "Custom..." Option hinzu
        dataset_options = [None] + datasets + ["Custom..."]
        dataset_selection = st.selectbox(
            "Dataset (optional)",
            options=dataset_options,
            format_func=lambda x: "(keine Auswahl)" if x is None else x,
            index=0,
            key="verify_dataset_select",
        )

        dataset = None
        if dataset_selection == "Custom...":
            dataset = st.text_input("Custom dataset name", value="", key="verify_dataset_custom")
            if not dataset or not dataset.strip():
                dataset = None
        elif dataset_selection is not None:
            dataset = dataset_selection

        if use_fallback_datasets:
            st.caption("⚠️ Using fallback options (API lists unavailable)")

        # Load Example Button (nur wenn Dataset ausgewählt)
        # WICHTIG: Button muss VOR den Textareas sein, damit Session State gesetzt werden kann
        if dataset and dataset != "Custom...":
            if st.button("📥 Load random example", key="verify_load_random_example"):
                try:
                    with st.spinner("Lade Beispiel..."):
                        article, summary, example_id, metadata = dataset_loader.load_random_example(
                            dataset, seed=None
                        )
                        if article and summary:
                            # Setze Werte DIREKT in die Textarea-Keys
                            st.session_state["article_text"] = article
                            st.session_state["summary_text"] = summary
                            # Speichere Original für Undo
                            st.session_state["summary_text_original"] = summary
                            st.session_state["loaded_example_meta"] = {
                                "dataset": dataset,
                                "example_id": example_id,
                                "metadata": metadata,
                            }
                            st.rerun()
                        else:
                            st.error(
                                f"❌ Konnte kein Beispiel aus '{dataset}' laden (leere Felder)."
                            )
                except FileNotFoundError as e:
                    st.error(f"❌ Dataset-Datei nicht gefunden: {e}")
                except ValueError as e:
                    st.error(f"❌ Fehler beim Laden: {e}")
                except Exception as e:
                    st.error(f"❌ Unerwarteter Fehler: {e}")

            # Zeige geladenes Beispiel-Info mit Ground Truth Badge
            if st.session_state.get("loaded_example_meta"):
                loaded = st.session_state["loaded_example_meta"]
                if loaded.get("dataset") == dataset:
                    # Ground Truth Badge
                    has_error = loaded.get("metadata", {}).get("has_error")
                    example_id = loaded.get("example_id", "N/A")

                    # Badge je nach has_error
                    if has_error is True:
                        st.warning(
                            f"📄 **Dataset:** {dataset} | **Example:** {example_id} | "
                            f"**Ground truth:** has_error=True ⚠️"
                        )
                    elif has_error is False:
                        st.success(
                            f"📄 **Dataset:** {dataset} | **Example:** {example_id} | "
                            f"**Ground truth:** has_error=False ✅"
                        )
                    else:
                        st.info(f"📄 **Dataset:** {dataset} | **Example:** {example_id}")

                    # Debug-Ausgabe nur bei Bedarf
                    if st.session_state.get("show_debug", False):
                        st.caption(
                            f"DEBUG: Article length={len(st.session_state['article_text'])}, "
                            f"Summary length={len(st.session_state['summary_text'])}"
                        )

        # LLM Model Dropdown (mit Caching)
        try:
            models = get_cached_models()
            use_fallback_models = False
        except Exception:
            models = api_client.DEFAULT_MODELS.copy()
            use_fallback_models = True

        # Füge "Custom..." Option hinzu
        model_options = [None] + models + ["Custom..."]
        model_selection = st.selectbox(
            "LLM Model (optional)",
            options=model_options,
            format_func=lambda x: "(keine Auswahl)" if x is None else x,
            index=0,
            key="verify_model_select_adv",
        )

        llm_model = None
        if model_selection == "Custom...":
            llm_model = st.text_input("Custom model", value="", key="verify_model_custom_adv")
            if not llm_model or not llm_model.strip():
                llm_model = None
        elif model_selection is not None:
            llm_model = model_selection

        if use_fallback_models:
            st.caption("⚠️ Using fallback options (API lists unavailable)")

        enable_explainability = st.checkbox(
            "Enable Explainability",
            value=True,
            key="verify_enable_explainability_adv",
            help="Kann Verifikation verlangsamen",
        )
        persist = st.checkbox(
            "Persist to DB",
            value=False,
            key="verify_persist_adv",
            help="Speichert Run in Postgres+Neo4j (kann Verifikation verlangsamen)",
        )
        run_llm_judge = st.checkbox(
            "Run LLM Judge (slow)",
            value=False,
            key="run_llm_judge",
            help="Adds extra LLM calls. May take longer.",
        )

    # Pending updates VOR Textarea-Rendering anwenden (kritisch!)
    if st.session_state["pending_article_text"] is not None:
        st.session_state["article_text"] = st.session_state["pending_article_text"]
        st.session_state["pending_article_text"] = None

    if st.session_state["pending_summary_text"] is not None:
        st.session_state["summary_text"] = st.session_state["pending_summary_text"]
        st.session_state["pending_summary_text"] = None

    # Textareas NACH Advanced Options - direkt an session_state gebunden
    col1, col2 = st.columns(2)

    with col1:
        # WICHTIG: Nur key= verwenden, kein value= Parameter!
        article_text = st.text_area(
            "Article Text",
            height=200,
            placeholder="Artikel-Text hier einfügen...",
            key="article_text",
        )

    with col2:
        # WICHTIG: Nur key= verwenden, kein value= Parameter!
        summary_text = st.text_area(
            "Summary Text",
            height=200,
            placeholder="Zusammenfassung hier einfügen...",
            key="summary_text",
        )

    # Debug Panel (optional, nur wenn show_debug aktiv)
    if st.session_state.get("show_debug", False):
        with st.expander("Debug Panel", expanded=False):
            st.caption("Debug-Informationen für Demo/Debugging")
            if article_text:
                # Normalisierter Text Preview
                from app.services.agents.factuality.number_normalization import (
                    normalize_text_for_number_extraction,
                )

                normalized_article = normalize_text_for_number_extraction(article_text[:300])
                normalized_summary = normalize_text_for_number_extraction(summary_text[:300])
                st.write("**Normalized Article Preview (first 300 chars):**")
                st.code(normalized_article, language=None)
                st.write("**Normalized Summary Preview (first 300 chars):**")
                st.code(normalized_summary, language=None)

    # Error-Inject Controls (nur wenn Summary vorhanden)
    if summary_text:
        st.divider()
        st.subheader("Error Injection (Demo)")
        st.caption("Injiziert gezielt Fehler für Demo-Tests. Verify bleibt manuell.")

        col_inj1, col_inj2, col_inj3 = st.columns(3)

        with col_inj1:
            inject_type = st.selectbox(
                "Inject error type",
                options=["None", "Factuality", "Coherence", "Readability"],
                key="inject_type",
                help="Wähle die Dimension für den Fehler",
            )

        with col_inj2:
            inject_severity = st.selectbox(
                "Inject severity",
                options=["low", "medium", "high"],
                index=1,  # default: medium
                key="inject_severity",
                help="Schweregrad des Fehlers",
            )

        with col_inj3:
            if st.button(
                "🔧 Inject selected error", key="btn_inject_selected", use_container_width=True
            ):
                if inject_type and inject_type != "None":
                    current = st.session_state.get("summary_text", "")
                    new_text, change_log = injector.inject(
                        current, inject_type.lower(), inject_severity
                    )
                    st.session_state["pending_summary_text"] = new_text
                    st.session_state["inject_change_log"] = change_log
                    st.rerun()
                else:
                    st.warning("Bitte wähle einen Error-Type aus.")

        # Change Log anzeigen
        if st.session_state.get("inject_change_log"):
            st.info(f"📝 {st.session_state['inject_change_log']}")

        # Undo Button (optional, P1)
        if st.session_state.get("summary_text_original") and st.session_state.get(
            "summary_text"
        ) != st.session_state.get("summary_text_original"):
            if st.button("↩️ Undo injection", key="btn_undo_inject"):
                st.session_state["pending_summary_text"] = st.session_state["summary_text_original"]
                st.session_state["inject_change_log"] = None
                st.rerun()

    # Ready-to-Verify Indikator (nur wenn beide Felder gefüllt)
    if article_text and summary_text:
        if st.session_state.get("loaded_example_meta"):
            loaded = st.session_state["loaded_example_meta"]
            st.success(f"✅ Ready to verify (Example: {loaded.get('example_id', 'N/A')})")
        else:
            st.info("ℹ️ Ready to verify")

        # Persist-Hinweis
        if st.session_state.get("verify_persist_adv", False):
            st.caption("💾 Run will be stored in Postgres + Neo4j")

    # Verify Button (einziger Trigger für API-Call)
    if st.button("Verify", type="primary", key="verify_btn"):
        # Guardrails: Prüfe ob Felder gefüllt sind
        if not article_text or not article_text.strip():
            st.error("❌ Bitte Article-Text eingeben.")
        elif not summary_text or not summary_text.strip():
            st.error("❌ Bitte Summary-Text eingeben.")
        else:
            with st.spinner("Verifikation läuft... (kann bis zu 2 Minuten dauern)"):
                meta = {}
                if not enable_explainability:
                    meta["skip_explainability"] = "true"
                if not persist:
                    meta["skip_persist"] = "true"

                # Dataset für Verify-Request (verwende dataset aus Advanced Options)
                dataset_value = dataset if dataset and dataset.strip() else None
                model_value = llm_model if llm_model and llm_model.strip() else None

                try:
                    # Judge-Parameter wenn aktiviert
                    judge_params = None
                    if run_llm_judge:
                        judge_params = {
                            "judge_mode": "primary",
                            "judge_n": 3,
                            "judge_temperature": 0.0,
                            "judge_aggregation": "majority",
                        }

                    result = api_client.verify(
                        article_text=article_text,
                        summary_text=summary_text,
                        dataset=dataset_value,
                        llm_model=model_value,
                        meta=meta if meta else None,
                        run_llm_judge=run_llm_judge,
                        judge_params=judge_params,
                    )
                except Exception as e:
                    # Timeout oder andere Netzwerk-Fehler
                    error_msg = str(e)
                    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        st.error("⏱️ **Timeout:** Verifikation dauerte länger als erwartet.")
                        st.warning(
                            "💡 **Tipps:**\n"
                            "- Deaktiviere 'Persist to DB' für schnellere Verifikation\n"
                            "- Deaktiviere 'Enable Explainability' falls nicht benötigt\n"
                            "- Die API verarbeitet möglicherweise noch im Hintergrund"
                        )
                        result = {"success": False, "error": error_msg}
                    else:
                        st.error(f"❌ **Fehler:** {error_msg}")
                        result = {"success": False, "error": error_msg}

                if result.get("success"):
                    data = result["data"]
                    st.success(f"✅ Verifikation erfolgreich (Run-ID: {data.get('run_id')})")

                    # Scores anzeigen
                    render.render_scores(data)

                    # Comparison Panel (optional, collapsed)
                    with st.expander("Comparison (optional)", expanded=False):
                        render.render_comparison(data, summary_text)

                    # Explainability anzeigen (explainability_enabled aus Checkbox-State)
                    render.render_explainability(
                        data, summary_text, explainability_enabled=enable_explainability
                    )
                else:
                    st.error(f"❌ Fehler: {result.get('error', 'Unknown error')}")


# Tab 2: Runs
with tab2:
    st.header("Runs (Postgres)")

    if not db_client.is_available():
        st.warning("⚠️ Postgres nicht verfügbar")
        st.info("""
        **Troubleshooting:**
        - Prüfe `POSTGRES_DSN` Environment Variable
        - Oder setze: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
        - Stelle sicher, dass Postgres läuft und erreichbar ist
        """)
    else:
        runs = db_client.get_latest_runs(limit=50)

        if not runs:
            st.info("Keine Runs gefunden.")
        else:
            import pandas as pd

            # Tabelle mit Selectbox für Run-Auswahl
            df = pd.DataFrame(runs)
            df_display = df[["run_id", "created_at", "status", "num_results"]].copy()
            df_display.columns = ["Run ID", "Created At", "Status", "Results"]

            st.dataframe(df_display, use_container_width=True)

            # Run-Auswahl für Details
            run_options = [f"{r['run_id']} ({r['created_at']})" for r in runs]
            selected_run_str = st.selectbox(
                "Run für Details auswählen", [""] + run_options, key="runs_select_run"
            )

            # Detailansicht
            if selected_run_str:
                selected_idx = run_options.index(selected_run_str)
                selected_run_id = runs[selected_idx]["run_id"]

                st.divider()
                st.subheader(f"Run Details: {selected_run_id}")

                details = db_client.get_run_details(selected_run_id)
                if details:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Status", details["status"])
                        st.metric("Type", details["run_type"])
                    with col2:
                        st.metric("Created", details["created_at"] or "N/A")
                        st.metric("Results Count", len(details["results"]))

                    # Results
                    if details["results"]:
                        st.subheader("Verification Results")
                        for res in details["results"]:
                            with st.expander(f"{res['dimension']} (Score: {res['score']:.3f})"):
                                st.write("**Explanation:**", res["explanation"] or "N/A")
                                if res["issue_spans"]:
                                    st.write("**Issue Spans:**", len(res["issue_spans"]))

                    # Explainability
                    if details["explainability"]:
                        st.subheader("Explainability Report")
                        exp = details["explainability"]["report_json"]
                        if isinstance(exp, dict):
                            findings = exp.get("findings", [])
                            st.write(f"**Findings:** {len(findings)}")
                            st.json(exp)
                    else:
                        st.info(
                            "Kein Explainability-Report in Postgres. Möglicherweise nur in Neo4j verfügbar."
                        )
                else:
                    st.error("Run-Details konnten nicht geladen werden.")


# Tab 3: Status
with tab3:
    st.header("Status Documents")
    st.caption("Markdown-Dokumente aus docs/status/")

    available_docs = docs_view.get_available_docs()

    doc_options = {title: filename for filename, title, exists in available_docs if exists}
    missing_docs = [title for filename, title, exists in available_docs if not exists]

    if doc_options:
        selected_doc = st.selectbox(
            "Dokument auswählen", list(doc_options.keys()), key="status_select_doc"
        )
        if selected_doc:
            docs_view.render_doc(doc_options[selected_doc])
    else:
        st.warning("Keine Dokumente gefunden.")

    if missing_docs:
        st.divider()
        st.caption("Nicht verfügbare Dokumente:")
        for title in missing_docs:
            st.caption(f"  - {title}")
