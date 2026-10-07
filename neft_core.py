"""Интерфейс NEFT: сохранённая модель Condition + прежняя компоновка Streamlit.

Запуск: python -m streamlit run app.py
Модель: models/condition_pipeline.joblib (локальный доверенный файл).
Обучение здесь не запускается. Вся предобработка уже находится внутри Pipeline.
"""
from __future__ import annotations

import csv
import io
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "condition_pipeline.joblib"
META_PATH = ROOT / "models" / "condition_metadata.json"
NUMERIC = [
    "Pipe_Size_mm", "Thickness_mm", "Max_Pressure_psi", "Temperature_C",
    "Corrosion_Impact_Percent", "Thickness_Loss_mm", "Material_Loss_Percent", "Time_Years",
]
CATEGORICAL = ["Material", "Grade"]
FEATURES = NUMERIC + CATEGORICAL
CLASS_LABELS = {"Normal": "Норма", "Moderate": "Умеренные отклонения", "Critical": "Критическое состояние"}
COLORS = {"Normal": "#45c89a", "Moderate": "#e2b55a", "Critical": "#ed6969"}
DECISIONS = {"Normal": "Плановый контроль", "Moderate": "Дополнительная диагностика", "Critical": "Проверка специалистом"}
RECOMMENDATIONS = {
    "Normal": "Модель отнесла параметры к нормальному состоянию. Сохраняйте установленный регламент контроля.",
    "Moderate": "Модель выявила умеренные отклонения. Передайте результаты на дополнительную техническую диагностику.",
    "Critical": "Модель отнесла параметры к критическому состоянию. Необходимо проверить измерения и передать результат ответственному специалисту.",
}
LABELS = {
    "Pipe_Size_mm": "Диаметр трубы, мм", "Thickness_mm": "Толщина стенки, мм",
    "Max_Pressure_psi": "Максимальное давление, psi", "Temperature_C": "Температура, °C",
    "Corrosion_Impact_Percent": "Влияние коррозии, %", "Thickness_Loss_mm": "Потеря толщины, мм",
    "Material_Loss_Percent": "Потеря материала, %", "Time_Years": "Срок эксплуатации, лет",
    "Material": "Материал", "Grade": "Марка материала",
}
MATERIAL_LABELS = {
    "Carbon Steel": "Углеродистая сталь", "Stainless Steel": "Нержавеющая сталь",
    "Fiberglass": "Стеклопластик", "HDPE": "Полиэтилен высокой плотности", "PVC": "ПВХ",
}

CSS = """
<style>
.stApp {background:#0a0f15;color:#f5f7fa;}
.block-container {max-width:1450px;padding-top:1.6rem;padding-bottom:2.5rem;}
[data-testid="stHeader"] {background:transparent;}
#MainMenu,footer {visibility:hidden;}
h1 {color:#f7f9fc!important;font-weight:750!important;letter-spacing:-.03em;line-height:1.15!important;font-size:2.05rem!important;}
h2,h3 {color:#eef3f8!important;font-weight:650!important;}
[data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] p {color:#d9e3ec;}
[data-testid="stCaptionContainer"] p {color:#aebdca!important;font-size:.86rem!important;}
[data-testid="stWidgetLabel"] p {font-size:.86rem!important;line-height:1.35!important;}
[data-testid="stForm"],[data-testid="stVerticalBlockBorderWrapper"] {
 background:#121a24;border:1px solid #263445!important;border-radius:16px!important;}
[data-testid="stForm"] {padding:18px;}
[data-testid="stNumberInput"] input,[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"]>div {background:#0d141c!important;color:#f8fafc!important;}
[data-testid="stNumberInput"] button {background:#1a2633!important;color:#eef4fa!important;}
[data-testid="stSlider"] {color:#d9e3ec;}
[data-testid="stMetric"] {background:#121a24;border:1px solid #263445;padding:15px 14px;border-radius:14px;min-height:114px;}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p {
 color:#aebdca!important;font-size:.76rem!important;line-height:1.3!important;white-space:normal!important;overflow:visible!important;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] * {
 color:#f6f8fb!important;font-size:1.12rem!important;font-weight:650!important;line-height:1.3!important;
 white-space:normal!important;overflow:visible!important;overflow-wrap:anywhere!important;text-overflow:clip!important;}
[data-testid="stFormSubmitButton"] button {
 width:100%;min-height:44px;border-radius:10px;border:1px solid #68afff;
 background:linear-gradient(180deg,#4d9fff,#357fd4);color:white!important;font-weight:700;}
[data-testid="stFormSubmitButton"] button p {color:white!important;}
[data-testid="stFormSubmitButton"] button:hover {border-color:#96c9ff;background:#3d8be2;}
[data-testid="stFileUploaderDropzone"] {background:#0f1720;border:1px dashed #3b5068;border-radius:12px;}
[data-testid="stFileUploaderDropzoneInstructions"] span {font-size:0;}
[data-testid="stFileUploaderDropzoneInstructions"] span::after {content:"Перетащите CSV-файл сюда";font-size:14px;color:#d7e0e9;}
[data-testid="stFileUploaderDropzoneInstructions"] small {font-size:0;}
[data-testid="stFileUploaderDropzoneInstructions"] small::after {content:"CSV, до 5 МБ";font-size:12px;color:#aebdca;}
[data-testid="stFileUploaderDropzone"] button {font-size:0;}
[data-testid="stFileUploaderDropzone"] button>div {display:none;}
[data-testid="stFileUploaderDropzone"] button::after {content:"Выбрать файл";font-size:14px;}
[data-testid="stDataFrame"] {border:1px solid #263445;border-radius:12px;overflow:hidden;}
[data-testid="stAlert"] p {color:inherit!important;}
hr {border-color:#223040;}
@media(max-width:700px){h1{font-size:1.6rem!important;}.block-container{padding:1.2rem 1rem;}}
</style>
"""


def load_pipeline(path: Path) -> Any:
    """Загружать только свой локальный файл модели, не пользовательские загрузки."""
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл модели: {path}")
    with warnings.catch_warnings():
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("В файле нет классификатора с predict_proba.")
    if set(map(str, model.classes_)) != set(CLASS_LABELS):
        raise ValueError("Модель должна предсказывать три класса Condition: Normal, Moderate, Critical.")
    if set(getattr(model, "feature_names_in_", [])) != set(FEATURES):
        raise ValueError("Схема модели не совпадает с десятью признаками из train_model.py.")
    return model


def categories_from_model(model: Any) -> dict[str, list[str]]:
    encoder = model.named_steps["preprocessing"].named_transformers_["categorical"]
    return {name: [str(v) for v in values] for name, values in zip(CATEGORICAL, encoder.categories_)}


def prepare_input(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Проверка входов. Condition/index игнорируются; целевой столбец не нужен."""
    if frame.empty:
        raise ValueError("В файле нет записей.")
    if len(frame) > 20000:
        raise ValueError("Разрешено не более 20 000 строк за один анализ.")
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("В CSV есть повторяющиеся названия столбцов.")
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError("В CSV отсутствуют столбцы: " + ", ".join(missing))
    result = frame.loc[:, FEATURES].copy()
    for name in NUMERIC:
        values = pd.to_numeric(result[name].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if bad.any():
            raise ValueError(f"{LABELS[name]}: пустые или некорректные числа в {int(bad.sum())} строках.")
        result[name] = values.astype(float)
    for name in CATEGORICAL:
        result[name] = result[name].astype(str).str.strip()
        bad = ~result[name].isin(categories[name])
        if bad.any():
            raise ValueError(f"{LABELS[name]}: значения отсутствуют в обучающей выборке. Проверьте написание в CSV.")
    if (result[["Pipe_Size_mm", "Thickness_mm"]] <= 0).any().any():
        raise ValueError("Диаметр и толщина стенки должны быть больше нуля.")
    nonnegative = [c for c in NUMERIC if c not in ("Temperature_C", "Pipe_Size_mm", "Thickness_mm")]
    if (result[nonnegative] < 0).any().any():
        raise ValueError("Давление, потери, коррозия и срок эксплуатации не могут быть отрицательными.")
    return result


def predict(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Порядок вероятностей берётся из classes_, а не из фиксированного индекса."""
    x = prepare_input(frame, categories_from_model(model))
    labels = model.predict(x)
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    if probabilities.shape != (len(x), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Модель вернула некорректные вероятности.")
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Выход модели не является распределением вероятностей.")
    return labels, pd.DataFrame(probabilities, columns=[str(c) for c in model.classes_], index=x.index)


def read_uploaded_csv(raw: bytes) -> pd.DataFrame:
    if not raw or len(raw) > 5 * 1024 * 1024:
        raise ValueError("Нужен непустой CSV-файл размером до 5 МБ.")
    encodings = ("utf-16",) if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "cp1251")
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось прочитать кодировку CSV. Сохраните его в UTF-8.")
    if not text.strip():
        raise ValueError("CSV-файл пуст.")
    try:
        separator = csv.Sniffer().sniff(text[:65536], delimiters=",;\t").delimiter
    except csv.Error:
        separator = max(",;\t", key=text.splitlines()[0].count)
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator, strict=True)
        header = [v.strip() for v in next(reader)]
        if any(not c for c in header) or len(set(header)) != len(header):
            raise ValueError("В CSV есть пустые или повторяющиеся заголовки.")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Неверное число полей в строке CSV {reader.line_num}.")
            rows.append(row)
            if len(rows) > 20000:
                raise ValueError("Разрешено не более 20 000 строк.")
    except (csv.Error, StopIteration) as exc:
        raise ValueError("Повреждённый CSV-файл.") from exc
    return pd.DataFrame(rows, columns=header)


def quality_notes(x: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    notes = []
    count = int(((x["Thickness_Loss_mm"] > x["Thickness_mm"]) | (x["Material_Loss_Percent"] > 100)).sum())
    if count:
        notes.append(f"Проверьте измерения: в {count} строках потеря толщины больше толщины стенки или потеря материала больше 100%. Значения не исправлялись автоматически.")
    mismatches = ((100 * x["Thickness_Loss_mm"] / x["Thickness_mm"] - x["Material_Loss_Percent"]).abs() > .02)
    if mismatches.any():
        notes.append("Потеря материала не совпадает с отношением потери толщины к толщине стенки. Проверьте смысл полей и единицы измерения.")
    outside = []
    for name, bounds in metadata.get("training_numeric_ranges", {}).items():
        if name in x and not x[name].between(bounds["min"], bounds["max"]).all():
            outside.append(LABELS.get(name, name))
    if outside:
        notes.append("За пределами значений обучающей выборки: " + "; ".join(outside) + ". Достоверность такой оценки не проверена.")
    return notes


def feature_importance(model: Any) -> pd.DataFrame:
    """Общая важность признаков леса/дерева; не SHAP отдельного объекта."""
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["Признак", "Важность"])
    transformed = model.named_steps["preprocessing"].get_feature_names_out()
    totals = dict.fromkeys(FEATURES, 0.0)
    for name, value in zip(transformed, classifier.feature_importances_):
        short = str(name).split("__", 1)[-1]
        for feature in FEATURES:
            if short == feature or (feature in CATEGORICAL and short.startswith(feature + "_")):
                totals[feature] += float(value)
                break
    return pd.DataFrame({"Признак": [LABELS[c] for c in FEATURES], "Важность": [totals[c] * 100 for c in FEATURES]}).sort_values("Важность")


def gauge_figure(probability: float, color: str) -> go.Figure:
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=probability,
        number={"suffix": "%", "font": {"size": 36, "color": "#f2f6fa"}},
        title={"text": "Критическое состояние", "font": {"size": 14, "color": "#b4c1cf"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#617489", "tickfont": {"color": "#a4b3c2"}},
               "bar": {"color": color, "thickness": .32}, "bgcolor": "#1e2c3c", "borderwidth": 0},
    ))
    figure.update_layout(height=270, margin=dict(l=25, r=25, t=45, b=10),
                         paper_bgcolor="rgba(0,0,0,0)", font={"family": "Segoe UI, Arial", "color": "#f2f6fa"})
    return figure


def bar_figure(frame: pd.DataFrame, x: str, y: str, axis_title: str) -> go.Figure:
    figure = go.Figure(go.Bar(x=frame[x], y=frame[y], orientation="h", marker={"color": "#4b9cff"},
                             hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>"))
    figure.update_layout(height=330, margin=dict(l=10, r=20, t=10, b=35),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font={"color": "#d7e0e9", "family": "Segoe UI, Arial"},
                         xaxis={"title": axis_title, "gridcolor": "#253444", "zeroline": False},
                         yaxis={"title": None, "automargin": True}, showlegend=False)
    return figure


