import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="Предиктивное обслуживание",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# СТИЛЬ
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0a0f15;
        color: #f5f7fa;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    /* ---------------- Заголовки ---------------- */

    h1 {
        color: #f7f9fc !important;
        font-weight: 750 !important;
        letter-spacing: -0.03em;
        line-height: 1.12;
        margin-bottom: 0.2rem;
    }

    h2,
    h3 {
        color: #eef3f8 !important;
        font-weight: 650 !important;
    }

    p,
    li,
    span {
        color: #d7e0e9;
    }

    /* ---------------- Подписи ---------------- */

    [data-testid="stCaptionContainer"] p {
        color: #aebdca !important;
        font-size: 0.92rem !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #d9e3ec !important;
        font-weight: 550 !important;
    }

    /* ---------------- Контейнеры ---------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #121a24;
        border: 1px solid #263445 !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }

    /* ---------------- Поля ввода ---------------- */

    [data-testid="stNumberInput"] input {
        background: #0d141c !important;
        color: #f8fafc !important;
        border-color: #31445a !important;
    }

    [data-testid="stNumberInput"] button {
        background: #1a2633 !important;
        color: #eef4fa !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        border-color: #70b7ff !important;
    }

    /* ---------------- Карточки метрик ---------------- */

    [data-testid="stMetric"] {
        background: #121a24;
        border: 1px solid #263445;
        padding: 15px 16px;
        border-radius: 14px;
        min-height: 108px;
    }

    [data-testid="stMetricLabel"] p {
        color: #aebdca !important;
        font-size: 0.78rem !important;
        line-height: 1.25 !important;

        white-space: normal !important;
    }

    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow: visible !important;
    }

    [data-testid="stMetricValue"] > div {
        color: #f6f8fb !important;

        font-size: 1.15rem !important;
        line-height: 1.2 !important;

        white-space: normal !important;

        overflow-wrap: anywhere !important;
        word-break: normal !important;
    }

    /* ---------------- Кнопка ---------------- */

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {

        width: 100%;
        min-height: 44px;

        border-radius: 10px;

        border: 1px solid #68afff;

        background:
            linear-gradient(
                180deg,
                #4d9fff 0%,
                #357fd4 100%
            );

        color: white !important;

        font-weight: 700;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {

        border-color: #96c9ff;

        background:
            linear-gradient(
                180deg,
                #62abff 0%,
                #3d8be2 100%
            );

        color: white !important;
    }

    /* ---------------- File uploader ---------------- */

    [data-testid="stFileUploaderDropzone"] {

        background: #0f1720;

        border: 1px dashed #3b5068;

        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #d7e0e9 !important;
    }

    /* ---------------- Таблица ---------------- */

    [data-testid="stDataFrame"] {

        border: 1px solid #263445;

        border-radius: 12px;

        overflow: hidden;
    }

    /* ---------------- Alerts ---------------- */

    [data-testid="stAlert"] p {
        color: inherit !important;
    }

    hr {
        border-color: #223040;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ШАПКА
# ============================================================

st.title(
    "Разработка предиктивной системы управления фондом "
    "скважин нефтегазодобывающего предприятия"
)

st.caption(
    "Тарасов И.Ю. · Колбаса Д.В. · Макарцев М.А."
)

st.write("")


# ============================================================
# ОСНОВНОЙ ЭКРАН
# ============================================================

left, right = st.columns(
    [0.95, 1.35],
    gap="large"
)


# ============================================================
# ПАРАМЕТРЫ ОБЪЕКТА
# ============================================================

with left:

    st.subheader("Параметры объекта")

    with st.container(border=True):

        with st.form("diagnostics_form"):

            col1, col2 = st.columns(2)

            with col1:

                pipe_size = st.number_input(
                    "Диаметр трубы",
                    min_value=1.0,
                    max_value=100.0,
                    value=12.0,
                    step=0.5
                )

                thickness = st.number_input(
                    "Толщина стенки",
                    min_value=0.1,
                    max_value=50.0,
                    value=8.5,
                    step=0.1
                )

                pressure = st.number_input(
                    "Максимальное давление",
                    min_value=0.0,
                    max_value=5000.0,
                    value=1200.0,
                    step=10.0
                )

                temperature = st.number_input(
                    "Температура, °C",
                    min_value=-50.0,
                    max_value=300.0,
                    value=82.0,
                    step=1.0
                )

            with col2:

                corrosion = st.slider(
                    "Влияние коррозии, %",
                    min_value=0,
                    max_value=100,
                    value=67
                )

                thickness_loss = st.number_input(
                    "Потеря толщины",
                    min_value=0.0,
                    max_value=20.0,
                    value=3.1,
                    step=0.1
                )

                material_loss = st.slider(
                    "Потеря материала, %",
                    min_value=0,
                    max_value=100,
                    value=40
                )

                years = st.number_input(
                    "Срок эксплуатации, лет",
                    min_value=0,
                    max_value=100,
                    value=15,
                    step=1
                )

            st.form_submit_button(
                "Выполнить диагностику"
            )


# ============================================================
# РАСЧЁТ РИСКА
# ============================================================

risk = (
    corrosion * 0.30
    + material_loss * 0.20
    + min(thickness_loss * 10, 100) * 0.20
    + min(max(temperature, 0) / 1.5, 100) * 0.10
    + min(pressure / 20, 100) * 0.10
    + min(years * 3, 100) * 0.10
)

risk = max(
    0.0,
    min(
        100.0,
        risk
    )
)


# ============================================================
# СТАТУС
# ============================================================

if risk < 35:

    condition = "Норма"

    decision = "Не требуется"

    color = "#45c89a"

    recommendation = (
        "Критических отклонений не обнаружено. "
        "Рекомендуется продолжить плановый мониторинг."
    )


elif risk < 70:

    condition = "Повышенный риск"

    decision = "Требуется контроль"

    color = "#e2b55a"

    recommendation = (
        "Рекомендуется провести дополнительную диагностику "
        "и усилить контроль технических параметров."
    )


else:

    condition = "Критический риск"

    decision = "Требуется обслуживание"

    color = "#ed6969"

    recommendation = (
        "Рекомендуется запланировать техническое обслуживание "
        "и проверить факторы с наибольшим вкладом в риск."
    )


# ============================================================
# РЕЗУЛЬТАТ ДИАГНОСТИКИ
# ============================================================

with right:

    st.subheader("Результат диагностики")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Техническое состояние",
            condition
        )

    with metric2:

        st.metric(
            "Вероятность обслуживания",
            f"{risk:.1f}%"
        )

    with metric3:

        st.metric(
            "Решение системы",
            decision
        )


    # ========================================================
    # ИНДИКАТОР
    # ========================================================

    with st.container(border=True):

        gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=risk,

                number={
                    "suffix": "%",

                    "font": {
                        "size": 36,
                        "color": "#f2f6fa"
                    }
                },

                title={
                    "text": "Индекс технического риска",

                    "font": {
                        "size": 14,
                        "color": "#b4c1cf"
                    }
                },

                gauge={

                    "axis": {

                        "range": [
                            0,
                            100
                        ],

                        "tickcolor": "#617489",

                        "tickfont": {
                            "color": "#a4b3c2"
                        }
                    },

                    "bar": {

                        "color": color,

                        "thickness": 0.32
                    },

                    "bgcolor": "#121a24",

                    "borderwidth": 0,

                    "steps": [

                        {
                            "range": [0, 35],
                            "color": "rgba(69,200,154,0.10)"
                        },

                        {
                            "range": [35, 70],
                            "color": "rgba(226,181,90,0.10)"
                        },

                        {
                            "range": [70, 100],
                            "color": "rgba(237,105,105,0.10)"
                        }
                    ]
                }
            )
        )


        gauge.update_layout(

            height=270,

            margin=dict(
                l=25,
                r=25,
                t=45,
                b=10
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font={

                "family":
                    "Segoe UI, Arial",

                "color":
                    "#f2f6fa"
            }
        )


        st.plotly_chart(

            gauge,

            use_container_width=True,

            config={
                "displayModeBar": False
            }
        )


    # ========================================================
    # РЕКОМЕНДАЦИЯ
    # ========================================================

    if risk < 35:

        st.success(
            recommendation
        )

    elif risk < 70:

        st.warning(
            recommendation
        )

    else:

        st.error(
            recommendation
        )


# ============================================================
# НИЖНИЙ БЛОК
# ============================================================

st.divider()

bottom_left, bottom_right = st.columns(
    [1.4, 0.75],
    gap="large"
)


# ============================================================
# ФАКТОРЫ РИСКА
# ============================================================

with bottom_left:

    st.subheader(
        "Основные факторы риска"
    )


    factors = pd.DataFrame(
        {

            "Фактор": [

                "Коррозия",

                "Потеря материала",

                "Потеря толщины",

                "Температура",

                "Давление",

                "Срок эксплуатации"
            ],

            "Вклад": [

                corrosion * 0.30,

                material_loss * 0.20,

                min(
                    thickness_loss * 10,
                    100
                ) * 0.20,

                min(
                    max(
                        temperature,
                        0
                    ) / 1.5,
                    100
                ) * 0.10,

                min(
                    pressure / 20,
                    100
                ) * 0.10,

                min(
                    years * 3,
                    100
                ) * 0.10
            ]
        }
    )


    factors = factors.sort_values(
        "Вклад",
        ascending=True
    )


    chart = go.Figure(

        go.Bar(

            x=factors["Вклад"],

            y=factors["Фактор"],

            orientation="h",

            marker={
                "color": "#4b9cff"
            },

            hovertemplate=(
                "%{y}<br>"
                "Вклад: %{x:.1f}"
                "<extra></extra>"
            )
        )
    )


    chart.update_layout(

        height=310,

        margin=dict(
            l=10,
            r=20,
            t=10,
            b=35
        ),

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "rgba(0,0,0,0)",

        font={
            "color": "#d7e0e9",
            "family": "Segoe UI, Arial"
        },

        xaxis={

            "title":
                "Вклад в оценку риска",

            "gridcolor":
                "#253444",

            "zeroline":
                False,

            "tickfont": {
                "color": "#aebdca"
            }
        },

        yaxis={

            "title":
                None,

            "tickfont": {
                "color": "#d7e0e9"
            }
        },

        showlegend=False
    )


    with st.container(
        border=True
    ):

        st.plotly_chart(

            chart,

            use_container_width=True,

            config={
                "displayModeBar": False
            }
        )


# ============================================================
# ПАКЕТНЫЙ АНАЛИЗ
# ============================================================

with bottom_right:

    st.subheader(
        "Пакетный анализ"
    )


    with st.container(
        border=True
    ):

        st.write(
            "Загрузите CSV-файл для анализа "
            "нескольких объектов."
        )


        uploaded_file = st.file_uploader(

            "CSV-файл",

            type=["csv"],

            label_visibility="collapsed"
        )


        if uploaded_file is not None:

            try:

                df = pd.read_csv(
                    uploaded_file
                )


                st.metric(
                    "Количество объектов",
                    len(df)
                )


                st.dataframe(

                    df.head(10),

                    use_container_width=True,

                    hide_index=True
                )


            except Exception as exc:

                st.error(
                    f"Не удалось прочитать файл: {exc}"
                )


        else:

            st.caption(
                "После загрузки здесь появятся "
                "объекты и результаты диагностики."
            )