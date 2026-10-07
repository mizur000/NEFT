"""NEFT: карта вышек, аналитические панели и диагностика сохранённой моделью.
Запуск: python -m streamlit run app.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neft_core import (
    ROOT, MODEL_PATH, META_PATH, CSS, FEATURES, NUMERIC, CATEGORICAL, CLASS_LABELS,
    COLORS, DECISIONS, RECOMMENDATIONS, LABELS, MATERIAL_LABELS, load_pipeline,
    categories_from_model, prepare_input, predict, quality_notes, feature_importance,
    gauge_figure, bar_figure, read_uploaded_csv,
)
from fleet import discover_source, make_fleet, bind_record, accept_event, FIELD_KEYS


def main() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="НЕФТЬ — мониторинг скважин", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Разработка предиктивной системы управления фондом скважин нефтегазодобывающего предприятия")
    st.caption("Тарасов И.Ю. · Колбаса Д.В. · Макарцев М.А.")

    @st.cache_resource(show_spinner=False)
    def cached_model(path: str, stamp: int):
        return load_pipeline(Path(path))

    @st.cache_data(show_spinner=False, max_entries=4)
    def cached_fleet(raw: bytes, stamp: str, metadata_json: str, _model: Any):
        return make_fleet(raw, _model, json.loads(metadata_json), stamp)

    try:
        model_stamp = MODEL_PATH.stat().st_mtime_ns
        model = cached_model(str(MODEL_PATH), model_stamp)
        categories = categories_from_model(model)
    except FileNotFoundError:
        st.error("Не найден models/condition_pipeline.joblib. Оставьте обученную модель в папке models.")
        st.code("python train_model.py data/kaggle1.csv", language="bash")
        st.stop()
    except Exception as exc:
        st.error("Не удалось загрузить модель. Запускайте сайт в том же окружении, где выполнялось обучение.")
        st.code(str(exc), language="text")
        st.caption("При другой версии scikit-learn повторите обучение в своём окружении. Сайт не переобучает модель автоматически.")
        st.stop()

    metadata = {}
    if META_PATH.is_file():
        try:
            metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            st.warning("Метаданные не прочитаны. Проверка обучающих диапазонов недоступна.")
    if st.session_state.get("model_stamp") != model_stamp:
        st.session_state["condition_result"] = None
        st.session_state["binding_token"] = None
        st.session_state["model_stamp"] = model_stamp

    # Карта вставлена строго перед параметрами объекта.
    registry = None
    selected_record = None
    try:
        uploaded_raw = st.session_state.get("fleet_upload")
        if uploaded_raw is None:
            path = discover_source(ROOT)
            source_name, raw = path.name, path.read_bytes()
        else:
            source_name = st.session_state["fleet_upload_name"]
            raw = uploaded_raw
        registry = cached_fleet(raw, str(model_stamp), json.dumps(metadata, sort_keys=True), model)
        records = {item["id"]: item for item in registry["records"]}
        selected_id = st.session_state.get("selected_well")
        if selected_id not in records:
            selected_id = next(iter(records))
            st.session_state["selected_well"] = selected_id
        acknowledged = st.session_state.get("acknowledged_signals", {})
        ack_ids = [well_id for well_id, signature in acknowledged.items()
                   if well_id in records and records[well_id]["signature"] == signature]
        component_path = ROOT / "assets" / "fleet_dashboard"
        if not (component_path / "index.html").is_file():
            raise FileNotFoundError("Не найдена assets/fleet_dashboard/index.html. Скопируйте папку assets из архива.")
        dashboard = components.declare_component("neft_fleet_dashboard", path=str(component_path))
        event = dashboard(
            records=registry["records"], selected_id=selected_id,
            acknowledged=ack_ids, has_geo=registry["has_geo"],
            generated_ids=registry["generated_ids"], source=source_name,
            key="neft_fleet_dashboard", default=None,
        )
        if accept_event(event, records, st.session_state):
            st.rerun()
        selected_record = records[selected_id]
        bind_record(st.session_state, selected_record,
                    (selected_id, registry["source_hash"], model_stamp),)
        actions = st.columns([1, 1, 3])
        if actions[0].button("Обновить данные", width="stretch",
                             help="Перечитать файл. Приложение само не генерирует новые показания."):
            cached_fleet.clear()
            st.session_state["binding_token"] = None
            st.rerun()
        if uploaded_raw is not None and actions[1].button("Вернуть исходный CSV", width="stretch"):
            del st.session_state["fleet_upload"]
            st.session_state["selected_well"] = None
            st.session_state["binding_token"] = None
            st.rerun()
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        st.error("Карта не построена: " + str(exc))
        st.session_state["condition_result"] = None
        st.session_state["binding_token"] = None

    st.divider()
    left, right = st.columns([.95, 1.35], gap="large")
    with left:
        st.subheader("Параметры объекта")
        if selected_record:
            st.caption("Выбрано: " + selected_record["name"] + ". Изменения в форме не перезаписывают реестр и сигналы карты.")
        editable = selected_record is None or selected_record["valid"]
        if not editable:
            st.warning("В записи некорректные входы. Исправьте CSV или выберите другую вышку. Значения другого объекта не подставляются.")
            st.dataframe(pd.DataFrame(
                {"Параметр": [LABELS[k] for k in FEATURES],
                 "Значение": [str(selected_record["inputs"][k]) for k in FEATURES]}
            ), hide_index=True, width="stretch")
        else:
            defaults = {
                "Pipe_Size_mm": 800.0, "Thickness_mm": 15.48, "Max_Pressure_psi": 300.0,
                "Temperature_C": 84.9, "Corrosion_Impact_Percent": 16.04,
                "Thickness_Loss_mm": 4.91, "Material_Loss_Percent": 31.72, "Time_Years": 2.0,
                "Material": categories["Material"][0], "Grade": categories["Grade"][0],
            }
            for name, value in defaults.items():
                st.session_state.setdefault(FIELD_KEYS[name], value)
            with st.form("diagnostics_form"):
                c1, c2 = st.columns(2)
                values = {}
                for col, names in [(c1, ["Pipe_Size_mm", "Thickness_mm", "Max_Pressure_psi", "Temperature_C", "Material"]),
                                   (c2, ["Corrosion_Impact_Percent", "Thickness_Loss_mm", "Material_Loss_Percent", "Time_Years", "Grade"])]:
                    with col:
                        for name in names:
                            if name in CATEGORICAL:
                                values[name] = st.selectbox(
                                    LABELS[name], categories[name], key=FIELD_KEYS[name],
                                    format_func=(lambda v: MATERIAL_LABELS.get(v, v)) if name == "Material" else str,
                                )
                            else:
                                min_value = .01 if name in ("Pipe_Size_mm", "Thickness_mm") else None if name == "Temperature_C" else 0.0
                                step = 1.0 if name in ("Pipe_Size_mm", "Max_Pressure_psi", "Time_Years") else .01
                                values[name] = st.number_input(LABELS[name], min_value=min_value,
                                                               step=step, key=FIELD_KEYS[name])
                submitted = st.form_submit_button("Выполнить диагностику", type="primary", width="stretch")
            if submitted:
                st.session_state["condition_result"] = None
                try:
                    x = prepare_input(pd.DataFrame([values]), categories)
                    labels, proba = predict(model, x)
                    st.session_state["condition_result"] = {
                        "label": str(labels[0]), "probabilities": proba.iloc[0].to_dict(),
                        "notes": quality_notes(x, metadata),
                        "well_id": selected_record["id"] if selected_record else "",
                        "origin": "manual",
                    }
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))

    result = st.session_state.get("condition_result")
    with right:
        st.subheader("Результат диагностики")
        if result and result.get("origin") == "manual":
            st.caption("Расчёт по введённым параметрам. Это не обновление измерений выбранной вышки.")
        a, b, c = st.columns(3)
        label = result.get("label") if result else None
        risk = result["probabilities"].get("Critical") if result else None
        a.metric("Техническое состояние", CLASS_LABELS.get(label, "Нет оценки"))
        b.metric("Оценка критического состояния", f"{risk*100:.1f}%" if risk is not None else "—",
                 help="Вероятность класса Critical из модели; не калиброванная вероятность аварии или ремонта.")
        c.metric("Решение системы", "Проверить данные" if result and result["notes"] else DECISIONS.get(label, "—"))
        with st.container(border=True):
            if risk is not None:
                st.plotly_chart(gauge_figure(risk*100, COLORS[label]), width="stretch", theme=None,
                                config={"displayModeBar": False})
            else:
                st.info("Выберите объект на карте или выполните диагностику.")
        if label:
            message = {"Normal": st.success, "Moderate": st.warning, "Critical": st.error}[label]
            message(RECOMMENDATIONS[label])
        if result:
            for note in result["notes"][:4]:
                st.warning(note)
            if len(result["notes"]) > 4:
                st.caption(f"Ещё замечаний: {len(result['notes'])-4}. Проверьте диапазоны исходных данных.")

    st.divider()
    chart_col, batch_col = st.columns([1.4, .8], gap="large")
    with chart_col:
        importance = feature_importance(model)
        st.subheader("Важность признаков модели")
        if not importance.empty:
            with st.container(border=True):
                st.plotly_chart(bar_figure(importance, "Важность", "Признак", "Общая важность, %"),
                                width="stretch", theme=None, config={"displayModeBar": False})
            st.caption("Общая важность при обучении. Не объяснение отдельного прогноза.")
        else:
            st.caption("Для сохранённого алгоритма общая важность признаков недоступна.")
    with batch_col:
        st.subheader("Загрузка реестра")
        with st.container(border=True):
            st.write("CSV с параметрами объектов. Можно анализировать записи и отобразить их на карте.")
            upload = st.file_uploader("CSV-файл", type=["csv"], label_visibility="collapsed")
            if upload is not None:
                try:
                    uploaded = cached_fleet(upload.getvalue(), str(model_stamp),
                                            json.dumps(metadata, sort_keys=True), model)
                    st.caption(f"Распознано записей: {len(uploaded['records'])}.")
                    if st.button("Показать на карте", type="primary", width="stretch"):
                        st.session_state["fleet_upload"] = upload.getvalue()
                        st.session_state["fleet_upload_name"] = upload.name
                        st.session_state["selected_well"] = None
                        st.session_state["binding_token"] = None
                        st.rerun()
                except (ValueError, TypeError, OSError) as exc:
                    st.error(str(exc))
            st.caption("Condition не требуется. Дополнительно: well_id, well_name, latitude, longitude.")
        with st.expander("О данных и интерпретации"):
            st.write(
                "Модель оценивает состояние образцов трубопроводов по десяти признакам. "
                "Привязка строк Kaggle к вышкам условная. Снимок CSV не является онлайн-телеметрией. "
                "Красный сигнал требует проверки ответственным специалистом, но не является "
                "автоматической командой останова. Приложение не отправляет команды оборудованию."
            )
            st.write("Отметки «Принят в работу» хранятся только в текущей сессии и не отключают сигналы.")


if __name__ == "__main__":
    main()
