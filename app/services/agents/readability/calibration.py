"""
Kalibrierung für Readability-Scores.

Variante A: Lineare Regression
- Fit: pred_cal = a * pred + b, clamped to [0,1]
- Robust, einfach, keine zusätzlichen Dependencies
- **Clipping:** Werte werden auf [0,1] geclippt (keine Rundung auf Buckets)

Variante B: Isotonic Regression (optional, benötigt sklearn)
- Fit: Monotone, stückweise konstante Funktion
- Kann nicht-lineare Beziehungen modellieren
- **Clipping:** Werte werden auf [0,1] geclippt (keine Rundung auf Buckets)
"""

import json
from pathlib import Path
from typing import Any, Optional

# Optional: Isotonic Regression (benötigt sklearn)
try:
    from sklearn.isotonic import IsotonicRegression

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def fit_linear_calibration(
    preds: list[float],
    gts: list[float],
) -> tuple[float, float]:
    """
    Fittet lineare Kalibrierung: pred_cal = a * pred + b, clamped to [0,1].

    Returns:
        (a, b): Koeffizienten für pred_cal = a * pred + b
    """
    if len(preds) != len(gts) or len(preds) == 0:
        raise ValueError("preds und gts müssen gleich lang und nicht leer sein")

    # Lineare Regression: y = a*x + b
    # a = Cov(x,y) / Var(x)
    # b = mean(y) - a * mean(x)
    n = len(preds)
    mean_pred = sum(preds) / n
    mean_gt = sum(gts) / n

    cov = sum((preds[i] - mean_pred) * (gts[i] - mean_gt) for i in range(n)) / n
    var_pred = sum((preds[i] - mean_pred) ** 2 for i in range(n)) / n

    if var_pred < 1e-10:
        # Keine Varianz in Predictions -> keine Kalibrierung möglich
        return 1.0, 0.0

    a = cov / var_pred
    b = mean_gt - a * mean_pred

    return a, b


def apply_calibration(pred: float, a: float, b: float) -> float:
    """
    Wendet Kalibrierung an: pred_cal = a * pred + b, clamped to [0,1].
    """
    calibrated = a * pred + b
    if calibrated < 0.0:
        return 0.0
    if calibrated > 1.0:
        return 1.0
    return calibrated


def save_calibration_params(
    a: float,
    b: float,
    n_samples: int,
    out_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Speichert Kalibrierungsparameter als JSON."""
    params = {
        "method": "linear",
        "a": a,
        "b": b,
        "n_samples": n_samples,
        "metadata": metadata or {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def fit_isotonic_calibration(
    preds: list[float],
    gts: list[float],
) -> "IsotonicRegression":
    """
    Fittet Isotonic Kalibrierung (monotone, stückweise konstante Funktion).

    Requires sklearn.

    Returns:
        Fitted IsotonicRegression model
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "sklearn ist nicht verfügbar. Bitte installieren: pip install scikit-learn"
        )

    if len(preds) != len(gts) or len(preds) == 0:
        raise ValueError("preds und gts müssen gleich lang und nicht leer sein")

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(preds, gts)
    return model


def apply_isotonic_calibration(pred: float, model: "IsotonicRegression") -> float:
    """
    Wendet Isotonic Kalibrierung an: pred_cal = model.predict([pred])[0], clamped to [0,1].

    Note: IsotonicRegression mit out_of_bounds="clip" clippt bereits auf [0,1],
    aber wir clippen defensiv nochmal.
    """
    calibrated = model.predict([pred])[0]
    if calibrated < 0.0:
        return 0.0
    if calibrated > 1.0:
        return 1.0
    return calibrated


def save_calibration_params(
    a: float | None = None,
    b: float | None = None,
    n_samples: int | None = None,
    out_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    method: str = "linear",
    isotonic_model: Optional["IsotonicRegression"] = None,
    isotonic_X: list[float] | None = None,
    isotonic_y: list[float] | None = None,
) -> None:
    """
    Speichert Kalibrierungsparameter als JSON.

    Für linear: a, b müssen angegeben sein.
    Für isotonic: isotonic_model muss angegeben sein, und isotonic_X/isotonic_y für Rekonstruktion.
    """
    if method == "linear":
        if a is None or b is None or n_samples is None or out_path is None:
            raise ValueError(
                "Für linear calibration müssen a, b, n_samples, out_path angegeben sein"
            )
        params = {
            "method": "linear",
            "a": a,
            "b": b,
            "n_samples": n_samples,
            "metadata": metadata or {},
        }
    elif method == "isotonic":
        if isotonic_model is None or n_samples is None or out_path is None:
            raise ValueError(
                "Für isotonic calibration müssen isotonic_model, n_samples, out_path angegeben sein"
            )
        # Speichere die Trainingsdaten für Rekonstruktion
        # IsotonicRegression benötigt die originalen Trainingsdaten zum Rekonstruieren
        if isotonic_X is None or isotonic_y is None:
            # Versuche aus dem Modell zu extrahieren
            if hasattr(isotonic_model, "X_") and hasattr(isotonic_model, "y_"):
                isotonic_X = (
                    isotonic_model.X_.tolist()
                    if hasattr(isotonic_model.X_, "tolist")
                    else list(isotonic_model.X_)
                )
                isotonic_y = (
                    isotonic_model.y_.tolist()
                    if hasattr(isotonic_model.y_, "tolist")
                    else list(isotonic_model.y_)
                )
            else:
                raise ValueError(
                    "Für isotonic calibration müssen isotonic_X und isotonic_y angegeben sein (Trainingsdaten)"
                )
        params = {
            "method": "isotonic",
            "X_": isotonic_X,
            "y_": isotonic_y,
            "n_samples": n_samples,
            "metadata": metadata or {},
        }
    else:
        raise ValueError(f"Unbekannte Methode: {method}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_calibration_params(calibration_path: Path) -> tuple[str, Any]:
    """
    Lädt Kalibrierungsparameter aus JSON.

    Returns:
        (method, params): method ist "linear" oder "isotonic", params sind die Parameter
    """
    with calibration_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    method = params.get("method", "linear")

    if method == "linear":
        a = float(params["a"])
        b = float(params["b"])
        return "linear", (a, b)
    if method == "isotonic":
        if not HAS_SKLEARN:
            raise ImportError(
                "sklearn ist nicht verfügbar. Bitte installieren: pip install scikit-learn"
            )
        # Rekonstruiere IsotonicRegression aus gespeicherten Trainingsdaten
        model = IsotonicRegression(out_of_bounds="clip")
        X_ = params.get("X_", [])
        y_ = params.get("y_", [])
        if not X_ or not y_:
            raise ValueError(
                "Isotonic calibration benötigt Trainingsdaten (X_ und y_) für Rekonstruktion"
            )
        if len(X_) != len(y_):
            raise ValueError(f"X_ und y_ müssen gleich lang sein: {len(X_)} != {len(y_)}")
        # Fitte das Modell mit den gespeicherten Trainingsdaten
        model.fit(X_, y_)
        return "isotonic", model
    raise ValueError(f"Unbekannte Kalibrierungsmethode: {method}")
