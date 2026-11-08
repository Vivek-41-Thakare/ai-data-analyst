import streamlit as st
import plotly.graph_objects as go

from core.data_engine import DataEngine
from core.llm import OllamaClient
from core.analyzer import Analyst


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# SIMPLE WHITE UI
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN APP
    ===================================================== */

    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        background-color: #ffffff !important;
    }


    /* =====================================================
       TEXT
    ===================================================== */

    p,
    span,
    label,
    li {
        color: #000000 !important;
    }

    h1,
    h2,
    h3,
    h4 {
        color: #000000 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #555555 !important;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dddddd !important;
    }

    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #dddddd !important;
    }


    /* =====================================================
       CSV UPLOADER
    ===================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px dashed #bbbbbb !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] section > div {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] section div {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
    }


    /* =====================================================
       BROWSE FILES BUTTON
    ===================================================== */

    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #999999 !important;
        border-radius: 6px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #f5f5f5 !important;
        border-color: #ff3030 !important;
        color: #ff3030 !important;
    }


    /* =====================================================
       NORMAL BUTTONS
    ===================================================== */

    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 7px !important;
    }

    .stButton > button:hover {
        background-color: #f5f5f5 !important;
        color: #ff3030 !important;
        border-color: #ff3030 !important;
    }


    /* =====================================================
       TABS
    ===================================================== */

    button[data-baseweb="tab"] {
        color: #555555 !important;
        background-color: #ffffff !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff3030 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #ff3030 !important;
    }


    /* =====================================================
       CHAT
    ===================================================== */

    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: none !important;
    }


    /* =====================================================
       CHAT INPUT
    ===================================================== */

    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #ff3030 !important;
    }


    /* =====================================================
       EXPANDERS
    ===================================================== */

    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 7px !important;
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
    }


    /* =====================================================
       DIVIDERS
    ===================================================== */

    hr {
        border-color: #dddddd !important;
    }


    /* =====================================================
       TOP HEADER
       No SVG styling here.
       This prevents our custom CSS from interfering
       with Streamlit/Plotly SVG elements.
    ===================================================== */

    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }

    [data-testid="stToolbar"] {
        background-color: #ffffff !important;
    }

    [data-testid="stToolbar"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    [data-testid="stToolbar"] button:hover {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }

    header button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    header button:hover {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }


    /* =====================================================
       TOP-RIGHT POPUP / THREE DOT MENU
    ===================================================== */

    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] > div {
        background-color: #ffffff !important;
    }

    div[role="menu"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #dddddd !important;
    }

    div[role="menu"] * {
        color: #000000 !important;
    }

    div[role="menuitem"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    div[role="menuitem"]:hover {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }

    div[data-baseweb="popover"] * {
        color: #000000 !important;
    }

    div[data-baseweb="popover"] button {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] button:hover {
        background-color: #f5f5f5 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_evidence_item(item):
    """
    Remove duplicate markdown bullet characters.
    """
    text = str(item).strip()

    while text.startswith(("-", "*", "•")):
        text = text[1:].strip()

    return text


def render_evidence(evidence):
    """
    Render evidence consistently.
    """
    if not evidence:
        return

    st.markdown("**How the answer was derived**")

    for item in evidence:
        cleaned = clean_evidence_item(item)

        if cleaned:
            st.markdown(f"- {cleaned}")


def render_chart(chart, chart_key):
    """
    Render a Plotly chart directly.

    No SVG conversion.
    No HTML conversion.
    No string rendering.
    """

    if chart is None:
        return

    try:

        if isinstance(chart, go.Figure):

            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False,
                },
                key=chart_key,
            )

        else:

            st.warning(
                f"Chart object is not a Plotly Figure: {type(chart)}"
            )

    except Exception as exc:

        st.warning(
            f"Could not render the chart: {exc}"
        )


def render_result(result, chart_key):
    """
    Render one analyst result.
    """

    # =====================================================
    # ANSWER
    # =====================================================

    answer = result.get("answer")

    if answer:
        st.markdown(answer)


    # =====================================================
    # EVIDENCE
    # =====================================================

    render_evidence(
        result.get("evidence")
    )


    # =====================================================
    # PANDAS CODE
    # =====================================================

    code = result.get("code")

    if code:

        with st.expander(
            "View generated Pandas code"
        ):

            st.code(
                code,
                language="python",
            )


    # =====================================================
    # SQL
    # =====================================================

    sql = result.get("sql")

    if sql:

        with st.expander(
            "View SQL"
        ):

            st.code(
                sql,
                language="sql",
            )


    # =====================================================
    # CHART
    # =====================================================

    render_chart(
        result.get("chart"),
        chart_key,
    )


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    st.session_state.engine = None


# =========================================================
# TITLE
# =========================================================

st.title("📊 AI Data Analyst")

st.caption(
    "Upload CSV data, ask questions in natural language, "
    "and get evidence-backed analysis."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Dataset")

    uploads = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if uploads:

        try:

            st.session_state.engine = (
                DataEngine.from_uploads(uploads)
            )

            st.success(
                f"{len(uploads)} file(s) loaded"
            )

            for name, df in (
                st.session_state.engine.datasets.items()
            ):

                st.write(
                    f"**{name}** — "
                    f"{len(df):,} rows × "
                    f"{len(df.columns)} columns"
                )

        except Exception as exc:

            st.error(
                f"Could not load the CSV: {exc}"
            )

    st.divider()

    st.subheader("Quick actions")

    summary_clicked = st.button(
        "Generate dataset summary",
        use_container_width=True,
    )

    anomaly_clicked = st.button(
        "Detect anomalies",
        use_container_width=True,
    )

    if (
        summary_clicked
        and st.session_state.engine
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": (
                    "Give me a business summary "
                    "of the uploaded dataset."
                ),
            }
        )

        st.rerun()

    if (
        anomaly_clicked
        and st.session_state.engine
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": (
                    "Detect anomalies in the dataset."
                ),
            }
        )

        st.rerun()


# =========================================================
# NO DATASET
# =========================================================

if not st.session_state.engine:

    st.info("👈 Upload a CSV to begin.")

    st.markdown(
        """
### Try questions like

- Which region generated the highest revenue?
- Show monthly sales trends.
- What are the top five customers?
- Which products are underperforming?
- Detect anomalies in the dataset.
- Generate SQL for this analysis.
"""
    )

    st.stop()


# =========================================================
# DATA ENGINE
# =========================================================

engine = st.session_state.engine


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "💬 Analyst",
        "🔎 Data Quality",
        "📁 Datasets",
    ]
)


# =========================================================
# DATA QUALITY
# =========================================================

with tab2:

    st.subheader("Data quality")

    st.dataframe(
        engine.quality_report(),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# DATASETS
# =========================================================

with tab3:

    for name, df in engine.datasets.items():

        st.subheader(name)

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# ANALYST
# =========================================================

with tab1:

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    for index, msg in enumerate(
        st.session_state.messages
    ):

        role = msg.get("role")

        with st.chat_message(role):

            st.markdown(
                msg.get("content", "")
            )

            # =================================================
            # ASSISTANT RESULT
            # =================================================

            if role == "assistant":

                render_evidence(
                    msg.get("evidence")
                )

                code = msg.get("code")

                if code:

                    with st.expander(
                        "View generated Pandas code"
                    ):

                        st.code(
                            code,
                            language="python",
                        )

                sql = msg.get("sql")

                if sql:

                    with st.expander(
                        "View SQL"
                    ):

                        st.code(
                            sql,
                            language="sql",
                        )

                render_chart(
                    msg.get("chart"),
                    chart_key=f"history-chart-{index}",
                )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask your data anything..."
    )


    # =====================================================
    # PROCESS QUESTION
    # =====================================================

    if question:

        # =================================================
        # SAVE USER QUESTION
        # =================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        # =================================================
        # SHOW USER MESSAGE
        # =================================================

        with st.chat_message("user"):

            st.markdown(
                question
            )


        # =================================================
        # CREATE ANALYST
        # =================================================

        analyst = Analyst(
            engine=engine,
            llm=OllamaClient(),
            history=st.session_state.messages[-8:],
        )


        # =================================================
        # GENERATE ANSWER
        # =================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing the data..."
            ):

                result = analyst.answer(
                    question
                )


            # =================================================
            # RENDER RESULT
            # =================================================

            render_result(
                result,
                chart_key="current-result-chart",
            )


        # =================================================
        # SAVE COMPLETE RESULT
        # =================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get(
                    "answer",
                    "",
                ),
                "evidence": result.get(
                    "evidence"
                ),
                "code": result.get(
                    "code"
                ),
                "sql": result.get(
                    "sql"
                ),
                "chart": result.get(
                    "chart"
                ),
            }
        )