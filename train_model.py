from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits

BASE_DIR = Path(__file__).resolve().parent
TARGET = "Condition"
CLASSES = ["Normal", "Moderate", "Critical"]
NUMERIC = [
    "Pipe_Size_mm", "Thickness_mm", "Max_Pressure_psi", "Temperature_C",
    "Corrosion_Impact_Percent", "Thickness_Loss_mm", "Material_Loss_Percent",
    "Time_Years",
]
CATEGORICAL = ["Material", "Grade"]
FEATURES = NUMERIC + CATEGORICAL
WEAR_COLUMNS = ["Thickness_Loss_mm", "Material_Loss_Percent"]
SEED = 42
FOLDS = 5
TEST_SIZE = 0.2
MAX_ROWS = 200_000
MAX_BYTES = 50 * 1024 * 1024
MODEL_NAMES = {
    "dummy": "Константный прогноз",
    "logistic": "Логистическая регрессия",
    "tree": "Дерево решений",
    "forest": "Случайный лес",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate schema/labels; never invent labels, impute or drop rows silently."""
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"Не найден CSV-файл: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("CSV больше 50 МБ.")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("CSV пуст.")
    encodings = ("utf-16",) if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "cp1251")
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось прочитать кодировку CSV.")
    if not text.strip():
        raise ValueError("CSV пуст.")
    try:
        sep = csv.Sniffer().sniff(text[:65536], delimiters=",;\t").delimiter
    except csv.Error:
        sep = max(",;\t", key=text.splitlines()[0].count)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=sep, strict=True)
    try:
        columns = [v.strip() for v in next(reader)]
        if any(not c for c in columns) or len(columns) != len(set(columns)):
            raise ValueError("Пустые или повторные заголовки CSV.")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(f"Неверное число полей, строка {reader.line_num}.")
            rows.append([v.strip() for v in row])
            if len(rows) > MAX_ROWS:
                raise ValueError("Слишком много записей: лимит 200000.")
    except (csv.Error, StopIteration) as exc:
        raise ValueError("Повреждённый CSV.") from exc
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise ValueError("В CSV нет записей.")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Нет нужных столбцов: " + ", ".join(missing))
    extra = [c for c in df.columns if c not in FEATURES + [TARGET]]
    df = df[FEATURES + [TARGET]].copy()
    for column in NUMERIC:
        values = pd.to_numeric(df[column].str.replace(",", ".", regex=False), errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if bad.any():
            raise ValueError(f"{column}: пропуски или некорректные числа ({int(bad.sum())}).")
        df[column] = values.astype(float)
    for column in CATEGORICAL:
        if df[column].str.lower().isin(["", "na", "nan", "null", "none"]).any():
            raise ValueError(f"{column}: есть пропуски.")
    invalid_labels = sorted(set(df[TARGET]) - set(CLASSES))
    if invalid_labels:
        raise ValueError(f"Condition: неизвестные метки {invalid_labels}.")
    counts = df[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    if (counts < 10).any():
        raise ValueError("Нужно не менее 10 записей каждого из трёх классов.")
    # Fail rather than putting identical feature rows in different splits.
    if df.duplicated(subset=FEATURES, keep=False).any():
        raise ValueError("Есть повторяющиеся входы. Разберите дубли до разделения на обучение и тест.")
    if (df[["Pipe_Size_mm", "Thickness_mm"]] <= 0).any().any():
        raise ValueError("Диаметр и толщина должны быть положительными.")
    nonnegative = ["Max_Pressure_psi", "Corrosion_Impact_Percent", *WEAR_COLUMNS, "Time_Years"]
    if (df[nonnegative] < 0).any().any():
        raise ValueError("Некорректные отрицательные значения давления, потерь или срока.")
    df.index = pd.RangeIndex(1, len(df) + 1, name="record_number")
    return df, {"source_name": path.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                "encoding": encoding, "delimiter": sep, "ignored_columns": extra}


def pipeline(estimator: Any, numeric: list[str] | None = None) -> Pipeline:
    """Only fit preprocessing inside the training folds; drop all other inputs."""
    numeric = NUMERIC if numeric is None else numeric
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocessing", preprocessor), ("classifier", estimator)])


def data_checks(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Descriptive checks only: do not convert these rules into training labels."""
    loss, thickness = df["Thickness_Loss_mm"], df["Thickness_mm"]
    flags = pd.DataFrame({
        "loss_exceeds_thickness": loss > thickness,
        "material_loss_above_100": df["Material_Loss_Percent"] > 100,
        "corrosion_outside_0_100": ~df["Corrosion_Impact_Percent"].between(0, 100),
    }, index=df.index)
    issues = df.join(flags).loc[flags.any(axis=1)]
    observed_rule = np.where(loss <= 2, "Normal", np.where(loss <= 5, "Moderate", "Critical"))
    rule_matches = int((observed_rule == df[TARGET]).sum())
    ratio_error = (100 * loss / thickness - df["Material_Loss_Percent"]).abs()
    checks = {
        "flagged_records": len(issues),
        "flag_counts": {k: int(v) for k, v in flags.sum().items()},
        "thickness_rule_matches": rule_matches,
        "thickness_rule_records": len(df),
        "material_loss_ratio_matches_at_0_01": int((ratio_error <= 0.01).sum()),
        "material_loss_ratio_max_absolute_error": float(ratio_error.max()),
        "class_thickness_loss_ranges": {
            c: {"min": float(loss[df[TARGET] == c].min()), "max": float(loss[df[TARGET] == c].max())}
            for c in CLASSES
        },
    }
    return checks, issues


def cv_scores(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    result = cross_validate(
        model, x, y,
        cv=StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED),
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=1, error_score="raise",
    )
    return {"cv_macro_f1": float(result["test_macro_f1"].mean()),
            "cv_macro_f1_std": float(result["test_macro_f1"].std()),
            "cv_balanced_accuracy": float(result["test_balanced_accuracy"].mean())}


def write_report(out: Path, m: dict[str, Any], comparisons: list[dict[str, Any]]) -> None:
    checks = m["data_checks"]
    report = [
        "# Обучение первой модели Condition",
        f"Файл: `{m['source_name']}`. SHA-256: `{m['source_sha256']}`.",
        f"Записей: **{m['records']}**. Обучение: **{m['train_records']}**, тест: **{m['test_records']}**. Строки не удалялись; CSV не перезаписывался.",
        "## 1. Что оценивается",
        "Класс текущего технического состояния: Normal, Moderate, Critical. Это не прогноз даты отказа и не оценка потребности в ремонте. Модель не подтверждает безопасность эксплуатации.",
        "## 2. Замечания к данным",
        f"Записей с флагами качества: **{checks['flagged_records']}**. Список: `data_quality_issues.csv`.",
        f"Потеря толщины больше толщины стенки: **{checks['flag_counts']['loss_exceeds_thickness']}**. Потеря материала >100%: **{checks['flag_counts']['material_loss_above_100']}**. Если потеря отсчитывается от указанной толщины, это противоречие. Нужно уточнить смысл полей; автоматически обрезать их до 100% нельзя.",
        f"Формула `100 * Thickness_Loss_mm / Thickness_mm` совпадает с Material_Loss_Percent с допуском 0,01 в **{checks['material_loss_ratio_matches_at_0_01']}** строках. Это наблюдаемая зависимость, а не доказательство происхождения данных.",
        f"Наблюдаемое правило: потеря толщины <=2: Normal; >2 и <=5: Moderate; >5: Critical. Совпадений: **{checks['thickness_rule_matches']} / {m['records']}**. Пороги не являются технологическими нормами; источник разметки не подтверждён.",
        "Поэтому высокие метрики с признаками износа могут означать лишь воспроизведение простой разметки. Решение оставить спорные строки принято явно для исходного сравнения; ничего не дорисовывается.",
        "## 3. Протокол проверки",
        "Разбиение 80/20 с сохранением долей классов, random_state=42. Модель выбирается по среднему macro-F1 на 5 фолдах только обучающей части. Масштабирование и кодирование обучаются внутри Pipeline в каждом фолде. Константный прогноз нужен как нижняя база сравнения. Тест не участвует в выборе модели.",
        "Это внутренняя проверка уже изученного набора, а не независимая проверка на новых объектах. Независимость строк не подтверждена; при появлении историй объектов нужно перейти на групповое или временное разбиение.",
        "| Модель | CV macro-F1 | Стандартное отклонение |\n|---|---:|---:|\n" + "\n".join(
            f"| {MODEL_NAMES[r['model']]} | {r['cv_macro_f1']:.4f} | {r['cv_macro_f1_std']:.4f} |" for r in comparisons),
        f"Контрольный эксперимент: случайный лес без Thickness_Loss_mm и Material_Loss_Percent: CV macro-F1 **{m['without_wear_control']['cv_macro_f1']:.4f}**. Это проверка зависимости от износа, не поиск причин и не прогноз во времени. В выборе основной модели этот вариант не участвует.",
        "## 4. Результат на тесте",
        f"Выбрана модель: **{MODEL_NAMES[m['selected_model']]}**. Accuracy: **{m['test_accuracy']:.4f}**, macro-F1: **{m['test_macro_f1']:.4f}**.",
        "| Класс | Precision | Recall | F1 | Записей |\n|---|---:|---:|---:|---:|\n" + "\n".join(
            f"| {c} | {m['classification_report'][c]['precision']:.4f} | {m['classification_report'][c]['recall']:.4f} | {m['classification_report'][c]['f1-score']:.4f} | {int(m['classification_report'][c]['support'])} |" for c in CLASSES),
        "Вероятности predict_proba не калибровались. Их нельзя выдавать за проверенную вероятность аварии или ремонта.",
        "## 5. Сохранённые файлы",
        "`models/condition_pipeline.joblib` содержит предобработку и классификатор. Модель обучена на обучающей части, без переобучения на тесте. Повторная загрузка проверена: прогнозы совпадают.",
        "В `reports/`: cv_results.csv, test_predictions.csv, confusion_matrix.csv, split.csv, data_quality_issues.csv, metrics.json, training_environment.txt. В models/condition_metadata.json — схема входа, метрики и версии библиотек.",
        "## 6. Дальнейший шаг",
        "Для интерфейса нужны все 10 признаков, включая Material и Grade. Диаметр и толщина передаются в мм, давление — в psi. Старые диапазоны формы нужно сверить; менять app.py этот скрипт не будет. Параллельно нужно разобрать разметку и получить внешние данные для реального прогноза отказов.",
        "## Методические источники",
        "https://scikit-learn.org/stable/common_pitfalls.html\n\nhttps://scikit-learn.org/stable/model_persistence.html",
    ]
    (out / "reports" / "training_report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")


def train(source: Path, out: Path) -> dict[str, Any]:
    source, out = source.resolve(), out.resolve()
    output_csv_names = ["cv_results.csv", "data_quality_issues.csv", "test_predictions.csv",
                        "confusion_matrix.csv", "split.csv"]
    if source in {(out / "reports" / name).resolve() for name in output_csv_names}:
        raise ValueError("Путь CSV совпадает с путём отчёта. Выберите другую папку --output-dir.")
    df, source_info = read_dataset(source)
    checks, issues = data_checks(df)
    x, y = df[FEATURES], df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    if int(y_train.value_counts().min()) < FOLDS:
        raise ValueError("Недостаточно примеров для 5 фолдов.")
    candidates = {
        "dummy": pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic": pipeline(LogisticRegression(max_iter=3000, random_state=SEED)),
        "tree": pipeline(DecisionTreeClassifier(max_depth=3, random_state=SEED)),
        "forest": pipeline(RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                    random_state=SEED, n_jobs=1)),
    }
    comparisons = []
    with threadpool_limits(limits=1):
        for name, candidate in candidates.items():
            print(f"Проверка модели: {MODEL_NAMES[name]}", flush=True)
            comparisons.append({"model": name, **cv_scores(candidate, x_train, y_train)})
        # Select on training CV, not by looking at held-out test scores.
        eligible = [r for r in comparisons if r["model"] != "dummy"]
        selected = max(eligible, key=lambda r: r["cv_macro_f1"])["model"]
        reduced_numeric = [c for c in NUMERIC if c not in WEAR_COLUMNS]
        control = pipeline(RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                  random_state=SEED, n_jobs=1), reduced_numeric)
        control_scores = cv_scores(control, x_train, y_train)
        model = clone(candidates[selected]).fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                "scipy": scipy.__version__, "joblib": joblib.__version__}
    metadata = {
        **source_info, "target": TARGET, "features": FEATURES,
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "classes": model.classes_.tolist(), "records": len(df),
        "train_records": len(x_train), "test_records": len(x_test),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "selected_model": selected, "selection_metric": "training CV macro-F1",
        "seed": SEED, "cv_folds": FOLDS, "test_fraction": TEST_SIZE,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": classification_report(y_test, pred, labels=CLASSES,
                                                         output_dict=True, zero_division=0),
        "without_wear_control": control_scores, "data_checks": checks,
        "versions": versions, "source_unchanged": sha256(source) == source_info["source_sha256"],
        "training_categories": {c: sorted(x_train[c].unique().tolist()) for c in CATEGORICAL},
        "training_numeric_ranges": {c: {"min": float(x_train[c].min()), "max": float(x_train[c].max())}
                                    for c in NUMERIC},
        "probabilities_calibrated": False,
    }
    if not metadata["source_unchanged"]:
        raise ValueError("Исходный CSV изменился во время запуска. Повторите обучение.")
    models, reports = out / "models", out / "reports"
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    model_path = models / "condition_pipeline.joblib"
    joblib.dump(model, model_path, compress=3)
    reloaded = joblib.load(model_path)  # Only the artifact just written by this script.
    if not np.array_equal(reloaded.predict(x_test), pred) or not np.allclose(reloaded.predict_proba(x_test), proba):
        raise ValueError("Повторная загрузка модели дала другие прогнозы.")
    pd.DataFrame(comparisons).to_csv(reports / "cv_results.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(reports / "data_quality_issues.csv", encoding="utf-8-sig")
    predictions = pd.DataFrame({"actual_condition": y_test, "predicted_condition": pred}, index=x_test.index)
    for position, label in enumerate(model.classes_):
        predictions[f"probability_{label}"] = proba[:, position]
    predictions.to_csv(reports / "test_predictions.csv", encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(
        reports / "confusion_matrix.csv", index_label="actual_vs_predicted", encoding="utf-8-sig")
    split = pd.DataFrame({"Condition": y, "split": "train"}, index=df.index)
    split.loc[x_test.index, "split"] = "test"
    split.to_csv(reports / "split.csv", encoding="utf-8-sig")
    text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (models / "condition_metadata.json").write_text(text, encoding="utf-8")
    (reports / "metrics.json").write_text(text, encoding="utf-8")
    env = "# Python " + platform.python_version() + "\n" + "\n".join(
        f"{name}=={version}" for name, version in versions.items() if name != "python") + "\n"
    (reports / "training_environment.txt").write_text(env, encoding="utf-8")
    if selected == "tree":
        transformed_names = model.named_steps["preprocessing"].get_feature_names_out().tolist()
        rules = export_text(model.named_steps["classifier"], feature_names=transformed_names)
        prefix = "# Thresholds below apply to standardized features, not to raw millimeters.\n"
        (reports / "tree_rules.txt").write_text(prefix + rules, encoding="utf-8")
    write_report(out, metadata, comparisons)
    return metadata


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Обучение и сохранение классификатора Condition.")
    parser.add_argument("csv_path", type=Path, nargs="?", default=BASE_DIR / "data" / "kaggle1.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    args = parser.parse_args()
    try:
        result = train(args.csv_path, args.output_dir)
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"Модель обучена: {MODEL_NAMES[result['selected_model']]}")
    print(f"Модель сохранена: {(args.output_dir / 'models' / 'condition_pipeline.joblib').resolve()}")
    print(f"Отчёт: {(args.output_dir / 'reports' / 'training_report.md').resolve()}")
    print(f"Тест: Accuracy={result['test_accuracy']:.4f}; macro-F1={result['test_macro_f1']:.4f}")
    print(f"Записей с замечаниями: {result['data_checks']['flagged_records']}.")
    print("Исходный CSV не изменён. Обучение завершено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
