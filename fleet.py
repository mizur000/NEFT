"""Реестр, проверка записей и данные для карты. Исходные файлы не изменяются."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from neft_core import (
    NUMERIC, CATEGORICAL, FEATURES, LABELS, CLASS_LABELS,
    categories_from_model, predict, read_uploaded_csv,
)

SIGNALS = {**CLASS_LABELS, "Unknown": "Проверить данные"}
DEFAULT_PATHS = [
    "data/wells.csv", "data/kaggle1.csv", "data/kaggle1(1).csv",
    "kaggle1.csv", "kaggle1(1).csv", "data/market_pipe_thickness_loss_dataset.csv",
]
FIELD_KEYS = {name: "diagnostics_" + name for name in FEATURES}


def discover_source(root: Path) -> Path:
    for relative in DEFAULT_PATHS:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Добавьте data/kaggle1.csv или реестр data/wells.csv.")


def make_fleet(raw: bytes, model: Any, metadata: dict, model_stamp: str = "") -> dict:
    """Каждая строка становится объектом реестра; метка Condition НЕ читается моделью.
    Идентификатор, если его нет в CSV, определяется номером записи. Координаты
    не генерируются. Без координат используется явно условная схема.
    """
    df = read_uploaded_csv(raw)
    if df.empty:
        raise ValueError("В реестре нет записей.")
    missing = [name for name in FEATURES if name not in df.columns]
    if missing:
        raise ValueError("Нет признаков: " + ", ".join(missing))
    n = len(df)
    generated_ids = "well_id" not in df.columns
    ids = [f"СКВ-{i+1:04d}" for i in range(n)] if generated_ids else df["well_id"].astype(str).str.strip().tolist()
    if any(not value for value in ids) or len(set(ids)) != n:
        raise ValueError("well_id должен быть непустым и уникальным для каждой записи.")
    names = df["well_name"].astype(str).str.strip().tolist() if "well_name" in df else ids
    x = df[FEATURES].copy()
    categories = categories_from_model(model)
    invalid = np.zeros(n, dtype=bool)
    notes: list[list[str]] = [[] for _ in range(n)]
    for name in NUMERIC:
        values = pd.to_numeric(x[name].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if name in ("Pipe_Size_mm", "Thickness_mm"):
            bad |= (values <= 0).to_numpy()
        elif name != "Temperature_C":
            bad |= (values < 0).to_numpy()
        invalid |= bad
        for i in np.flatnonzero(bad):
            notes[i].append(LABELS[name] + ": некорректное или отсутствующее число.")
        x[name] = values.astype(float)
    for name in CATEGORICAL:
        x[name] = x[name].astype(str).str.strip()
        bad = ~x[name].isin(categories[name]).to_numpy()
        invalid |= bad
        for i in np.flatnonzero(bad):
            notes[i].append(LABELS[name] + ": значение неизвестно модели.")

    check_masks = [
        (x["Thickness_Loss_mm"] > x["Thickness_mm"], "Потеря толщины больше толщины стенки."),
        (x["Material_Loss_Percent"] > 100, "Потеря материала превышает 100%."),
        (~x["Corrosion_Impact_Percent"].between(0, 100), "Коррозия вне диапазона 0–100%."),
        ((100*x["Thickness_Loss_mm"]/x["Thickness_mm"] - x["Material_Loss_Percent"]).abs() > .02,
         "Потеря материала не согласуется с отношением потери толщины к толщине стенки."),
    ]
    for mask, note in check_masks:
        for i in np.flatnonzero(mask.to_numpy() & ~invalid):
            notes[i].append(note)
    for name, bounds in metadata.get("training_numeric_ranges", {}).items():
        if name not in NUMERIC:
            continue
        mask = ~x[name].between(bounds["min"], bounds["max"])
        for i in np.flatnonzero(mask.to_numpy() & ~invalid):
            notes[i].append(LABELS[name] + ": за диапазоном обучающей выборки.")

    predictions = {}
    good = x.loc[~invalid]
    if len(good):
        labels, proba = predict(model, good)
        for pos, i in enumerate(good.index):
            predictions[int(i)] = (str(labels[pos]), {str(k): float(v) for k, v in proba.iloc[pos].items()})

    # Не заменяем поврежденные реальные координаты выдуманными.
    has_geo = "latitude" in df and "longitude" in df
    if ("latitude" in df) != ("longitude" in df):
        raise ValueError("Для координат нужны оба столбца: latitude и longitude.")
    if has_geo:
        geo = df[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
        if (not np.isfinite(geo.to_numpy()).all()
                or not geo["latitude"].between(-90, 90).all()
                or not geo["longitude"].between(-180, 180).all()):
            raise ValueError("Некорректные координаты latitude/longitude.")
    records = []
    for i in range(n):
        label, probabilities = predictions.get(i, (None, {}))
        inputs = {key: float(x.at[i, key]) if key in NUMERIC else str(x.at[i, key]) for key in FEATURES}
        # JSON не допускает NaN; нет подстановки здоровых значений.
        inputs = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in inputs.items()}
        signal = "Unknown" if notes[i] or label is None else label
        signature = hashlib.sha256(json.dumps(
            [ids[i], inputs, signal, model_stamp], ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest()
        records.append({
            "id": ids[i], "name": names[i], "row": i+1,
            "label": label, "signal": signal, "probabilities": probabilities,
            "p_critical": probabilities.get("Critical"), "inputs": inputs,
            "notes": notes[i], "valid": not bool(invalid[i]), "signature": signature,
            "latitude": float(geo.iloc[i]["latitude"]) if has_geo else None,
            "longitude": float(geo.iloc[i]["longitude"]) if has_geo else None,
        })
    return {
        "records": records, "generated_ids": generated_ids, "has_geo": has_geo,
        "source_hash": hashlib.sha256(raw).hexdigest(),
        "counts": {s: sum(r["signal"] == s for r in records) for s in SIGNALS},
    }


def bind_record(state: Any, record: dict, token: tuple) -> None:
    """До создания виджетов: новая выбранная вышка обновляет форму и результат."""
    if state.get("binding_token") == token:
        return
    state["binding_token"] = token
    state["condition_result"] = {
        "label": record["label"], "probabilities": dict(record["probabilities"]),
        "notes": list(record["notes"]), "well_id": record["id"], "origin": "registry",
    }
    if record["valid"]:
        for name, value in record["inputs"].items():
            state[FIELD_KEYS[name]] = value


def accept_event(event: object, records: dict, state: Any) -> bool:
    """События проходят проверку id; повторный ответ компонента не выполняется дважды."""
    if not isinstance(event, dict):
        return False
    event_id = event.get("nonce")
    if not isinstance(event_id, str) or not event_id or state.get("last_fleet_event") == event_id:
        return False
    state["last_fleet_event"] = event_id
    record = records.get(str(event.get("id", "")))
    if record is None:
        return False
    action = event.get("action")
    if action == "select":
        state["selected_well"] = record["id"]
        state["binding_token"] = None
        return True
    if action == "ack" and record["signal"] != "Normal":
        ack = dict(state.get("acknowledged_signals", {}))
        ack[record["id"]] = record["signature"]
        state["acknowledged_signals"] = ack
        return True
    return False
