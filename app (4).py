"""
Class Participation Evaluation Dashboard
=========================================
A Streamlit app that parses Zoom/Google Meet transcripts (.vtt or .txt),
sends them to the Groq API (LLaMA-3.3-70b-versatile), and renders
interactive dashboards for professors and students.

Setup:
    1. pip install streamlit groq plotly
    2. Add your Groq API key to .streamlit/secrets.toml:
           [secrets]
           GROQ_API_KEY = "gsk_..."
    3. Run: streamlit run app.py
"""

import json
import re
import streamlit as st
from groq import Groq
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Class Participation Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS — vibrant blue & white theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:        #0a1628;
    --bg-mid:         #0d1e3d;
    --bg-card:        #102048;
    --bg-card-hover:  #163060;
    --accent-cyan:    #38d9f5;
    --accent-yellow:  #ffd166;
    --accent-green:   #06d6a0;
    --accent-red:     #ff6b6b;
    --text-primary:   #ffffff;
    --text-secondary: #a8c4e0;
    --border:         #1e3a6e;
    --border-bright:  #2a52a0;
    --radius:         12px;
}

/* ── Full page blue background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container,
[data-testid="stMainBlockContainer"] {
    background: linear-gradient(160deg, #0a1628 0%, #0d1e3d 50%, #0a1e45 100%) !important;
    color: #ffffff !important;
}

/* ── Global typography ── */
html, body, [class*="css"], p, li, span, div {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary);
}
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    color: #ffffff !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061022 0%, #0a1830 100%) !important;
    border-right: 2px solid var(--border-bright) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent-cyan) !important;
    font-size: 1.05rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: .4rem;
}
[data-testid="stSidebar"] label { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] small { color: var(--text-secondary) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] * { color: #ffffff !important; }

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid var(--border-bright) !important;
    color: #ffffff !important;
    border-radius: var(--radius) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-secondary) !important; }
[data-testid="stTextInput"] label { color: var(--text-secondary) !important; font-size: .85rem; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(16, 32, 72, 0.9) !important;
    border: 1.5px solid var(--border-bright) !important;
    color: #ffffff !important;
    border-radius: var(--radius) !important;
}
[data-testid="stSelectbox"] label { color: var(--text-secondary) !important; }
[data-testid="stSelectbox"] svg   { fill: var(--accent-cyan) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #102048, #163060) !important;
    border: 1.5px solid var(--border-bright) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem !important;
    transition: all .25s ease;
    box-shadow: 0 4px 20px rgba(56,217,245,0.08);
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 6px 28px rgba(56,217,245,0.2) !important;
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: .8rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { color: var(--accent-green) !important; }

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; }
[data-testid="stAlert"] {
    background: rgba(16,32,72,0.8) !important;
    border-color: var(--border-bright) !important;
}
[data-testid="stAlert"] * { color: #ffffff !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: linear-gradient(135deg, #102048, #0d1e3d) !important;
    border: 1.5px solid var(--border-bright) !important;
    border-radius: var(--radius) !important;
    margin-bottom: .6rem;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stExpander"]:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 4px 18px rgba(56,217,245,0.14) !important;
}
[data-testid="stExpander"] summary { color: #ffffff !important; font-weight: 500; }
[data-testid="stExpander"] * { color: #ffffff !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: 1.5px solid #42a5f5 !important;
    border-radius: var(--radius) !important;
    padding: .6rem 1.8rem !important;
    font-size: .95rem !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4) !important;
    letter-spacing: .03em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 6px 24px rgba(56,217,245,0.35) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid var(--border-bright);
    background: transparent;
    gap: .5rem;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text-secondary) !important;
    padding: .5rem 1.2rem .7rem !important;
    font-size: 1rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all .2s;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    background: rgba(56,217,245,0.08) !important;
    color: #ffffff !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 3px solid var(--accent-cyan) !important;
    background: rgba(56,217,245,0.06) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1.5px solid var(--border-bright) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] * { color: #ffffff !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: var(--accent-cyan) !important; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0d2060 0%, #0a1628 70%);
    border: 1.5px solid var(--border-bright);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 8px 40px rgba(56,217,245,0.1);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,217,245,0.14), transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 200px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(255,209,102,0.08), transparent 70%);
    border-radius: 50%;
}
.hero-header .icon {
    font-size: 3.2rem;
    filter: drop-shadow(0 0 14px rgba(56,217,245,0.55));
    z-index: 1;
}
.hero-header h1 {
    margin: 0;
    font-size: 2rem;
    color: #ffffff !important;
    z-index: 1;
}
.hero-header p {
    margin: .35rem 0 0;
    color: var(--text-secondary);
    font-size: .95rem;
    z-index: 1;
}

/* ── Insight card ── */
.insight-card {
    background: linear-gradient(135deg, #102048, #0d1e3d);
    border: 1.5px solid var(--border-bright);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: box-shadow .2s, border-color .2s;
}
.insight-card:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 6px 28px rgba(56,217,245,0.14);
}
.insight-card .label {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--accent-cyan);
    margin-bottom: .5rem;
    font-weight: 600;
}
.insight-card .value {
    color: #ffffff;
    line-height: 1.7;
    font-size: .95rem;
}

/* ── Section heading accent ── */
.section-title {
    color: #ffffff !important;
    border-left: 4px solid var(--accent-cyan);
    padding-left: .75rem;
    margin: 1.5rem 0 .8rem;
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
}

/* ── Tag pill ── */
.tag {
    display: inline-block;
    background: rgba(255,209,102,0.15);
    color: var(--accent-yellow);
    border: 1px solid rgba(255,209,102,0.35);
    border-radius: 20px;
    padding: .25rem .85rem;
    font-size: .82rem;
    margin: .2rem .18rem;
    font-weight: 500;
    transition: background .2s;
}
.tag:hover { background: rgba(255,209,102,0.28); }

/* ── Answered-by chips ── */
.student-chip {
    display: inline-block;
    background: rgba(6,214,160,0.12);
    color: var(--accent-green);
    border: 1px solid rgba(6,214,160,0.3);
    border-radius: 20px;
    padding: .22rem .8rem;
    font-size: .83rem;
    margin: .15rem;
    font-weight: 500;
}

/* ── Feedback rows ── */
.feedback-row {
    background: linear-gradient(135deg, #102048, #0d1e3d);
    border-left: 4px solid var(--accent-yellow);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    box-shadow: 0 3px 14px rgba(0,0,0,0.2);
}
.feedback-row.strength { border-left-color: var(--accent-green); }
.feedback-row.weakness { border-left-color: var(--accent-red); }
.feedback-row .fb-label {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text-secondary);
    margin-bottom: .35rem;
    font-weight: 600;
}
.feedback-row .fb-text {
    color: #ffffff;
    line-height: 1.7;
    font-size: .93rem;
}

/* ── Divider ── */
hr { border-color: var(--border-bright) !important; opacity: .5; }

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }
small, .stCaption { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def get_groq_client() -> Groq:
    """Initialise Groq client from Streamlit secrets or environment."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "⚠️ Groq API key not found. "
            "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or set it as an environment variable."
        )
        st.stop()
    return Groq(api_key=api_key)


def parse_transcript(raw_text: str) -> str:
    """
    Strip VTT/SRT formatting from a transcript.
    Keeps speaker-prefixed lines like 'Alice: Hello class.'
    """
    text = re.sub(r"^WEBVTT.*?\n\n", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SYSTEM_PROMPT_TEMPLATE = """
You are an expert educational analyst. You will receive a transcript from an online class session.
Your task is to evaluate student participation and map student answers to professor questions.

The professor in this session is: {professor_name}

Carefully read the transcript and return ONLY a valid JSON object (no markdown fences, no explanation)
with EXACTLY this structure:

{{
  "professor_dashboard": {{
    "overall_class_understanding": "<2-4 sentence summary of how well the class grasped the material>",
    "topics_to_review": ["<topic 1>", "<topic 2>"],
    "teaching_feedback": "<Specific, constructive advice for the professor to improve future sessions>",
    "question_mapping": [
      {{
        "professor_question": "<exact or paraphrased question the professor asked>",
        "students_who_answered": ["<Student Name>", "..."]
      }}
    ]
  }},
  "student_evaluations": [
    {{
      "name": "<Student Name>",
      "scores": {{
        "relevance": <integer 1-10>,
        "knowledgeability": <integer 1-10>,
        "engagement": <integer 1-10>
      }},
      "feedback": {{
        "strengths": "<What the student did well>",
        "weaknesses": "<Where the student fell short>",
        "needs_improvement": "<One or two actionable steps the student should take>"
      }}
    }}
  ]
}}

Important rules:
- Only include students who actually spoke in the transcript (exclude the professor).
- Scores must be integers between 1 and 10.
- topics_to_review should contain concepts students seemed confused about.
- Return ONLY the JSON object — no markdown, no preamble.
"""


def call_groq(transcript: str, professor_name: str) -> dict:
    """Send the transcript to Groq and return the parsed JSON dict."""
    client = get_groq_client()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        professor_name=professor_name if professor_name else "the professor"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Here is the class transcript:\n\n{transcript}"},
        ],
        temperature=0.3,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"The LLM returned malformed JSON: {exc}")
        with st.expander("Raw LLM output (for debugging)"):
            st.code(raw, language="json")
        st.stop()


def score_color(score: int) -> str:
    if score >= 8:  return "#06d6a0"   # green  – great
    if score >= 5:  return "#ffd166"   # yellow – okay
    return "#ff6b6b"                   # red    – needs work


def radar_chart(scores: dict, name: str) -> go.Figure:
    categories    = ["Relevance", "Knowledgeability", "Engagement"]
    values        = [scores["relevance"], scores["knowledgeability"], scores["engagement"]]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(56,217,245,0.18)",
        line=dict(color="#38d9f5", width=2.5),
        name=name,
        hovertemplate="%{theta}: <b>%{r}/10</b><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1e3d",
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(color="#a8c4e0", size=10),
                gridcolor="#1e3a6e",
                linecolor="#1e3a6e",
            ),
            angularaxis=dict(
                tickfont=dict(color="#ffffff", size=13, family="DM Serif Display"),
                gridcolor="#1e3a6e",
                linecolor="#1e3a6e",
            ),
        ),
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0a1628",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
        height=340,
    )
    return fig


# ─────────────────────────────────────────────
#  UI — Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Setup")
    st.markdown("---")

    professor_name = st.text_input(
        "Professor's Name",
        placeholder="e.g. Dr. Sarah Chen",
        help="Used to distinguish the professor's speech from students in the transcript.",
    )

    uploaded_file = st.file_uploader(
        "Upload Transcript",
        type=["vtt", "txt"],
        help="Supports Zoom / Google Meet .vtt exports or plain-text transcripts with speaker labels.",
    )

    analyse_btn = st.button("⚡ Analyse Transcript", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#a8c4e0'>Powered by Groq · LLaMA 3.3 70B<br>"
        "Built with Streamlit · Plotly</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Hero header
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="icon">🏛️</div>
    <div>
        <h1>Class Participation Evaluator</h1>
        <p>Upload a class transcript · Get AI-powered participation insights in seconds</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  State management
# ─────────────────────────────────────────────

if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ─────────────────────────────────────────────
#  Analysis trigger
# ─────────────────────────────────────────────

if analyse_btn:
    if uploaded_file is None:
        st.warning("Please upload a transcript file first.")
    elif not professor_name.strip():
        st.warning("Please enter the professor's name so the AI can identify them in the transcript.")
    else:
        raw_text  = uploaded_file.read().decode("utf-8", errors="replace")
        clean_txt = parse_transcript(raw_text)

        with st.spinner("🔍 Sending transcript to Groq · LLaMA 3.3 70B — this may take 15–40 seconds…"):
            st.session_state.analysis = call_groq(clean_txt, professor_name.strip())

        st.success("✅ Analysis complete! Explore the dashboards below.")

# ─────────────────────────────────────────────
#  Dashboard tabs
# ─────────────────────────────────────────────

if st.session_state.analysis is None:
    st.info(
        "👈  Upload a transcript and click **Analyse Transcript** to get started.\n\n"
        "**Supported formats:** Zoom `.vtt` exports, Google Meet `.txt` transcripts, "
        "or any plain-text file with `Speaker Name: dialogue` formatting."
    )
    st.stop()

data       = st.session_state.analysis
prof_dash  = data.get("professor_dashboard", {})
student_ev = data.get("student_evaluations", [])

tab_prof, tab_student = st.tabs(["📋  Professor Dashboard", "🎓  Student Dashboard"])

# ════════════════════════════════════════════
#  TAB 1 — Professor Dashboard
# ════════════════════════════════════════════

with tab_prof:

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="insight-card">
            <div class="label">📊 Overall Class Understanding</div>
            <div class="value">{}</div>
        </div>
        """.format(prof_dash.get("overall_class_understanding", "N/A")), unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="insight-card">
            <div class="label">💡 Teaching Feedback</div>
            <div class="value">{}</div>
        </div>
        """.format(prof_dash.get("teaching_feedback", "N/A")), unsafe_allow_html=True)

    # — Topics to review
    topics = prof_dash.get("topics_to_review", [])
    if topics:
        st.markdown('<div class="section-title">🔖 Topics to Review</div>', unsafe_allow_html=True)
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in topics)
        st.markdown(f'<div style="margin-bottom:1.6rem">{tags_html}</div>', unsafe_allow_html=True)

    # — Question mapping
    qmapping = prof_dash.get("question_mapping", [])
    if qmapping:
        st.markdown('<div class="section-title">❓ Question Mapping</div>', unsafe_allow_html=True)
        st.caption("Each question asked by the professor, with the students who responded.")

        for i, item in enumerate(qmapping, 1):
            question  = item.get("professor_question", f"Question {i}")
            answerers = item.get("students_who_answered", [])

            with st.expander(f"Q{i}: {question}"):
                if answerers:
                    chips = "".join(
                        f'<span class="student-chip">👤 {s}</span>' for s in answerers
                    )
                    st.markdown(
                        f'<div style="padding:.3rem 0">'
                        f'<span style="color:#a8c4e0;font-size:.8rem;margin-right:.5rem">Answered by:</span>'
                        f'{chips}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span style="color:#ff6b6b;font-size:.9rem">⚠️ No students answered this question.</span>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No question mapping data was returned by the model.")

    # — Quick participation table
    if student_ev:
        st.markdown('<div class="section-title">📈 Quick Participation Summary</div>', unsafe_allow_html=True)
        rows = []
        for s in student_ev:
            sc  = s.get("scores", {})
            avg = round((sc.get("relevance", 0) + sc.get("knowledgeability", 0) + sc.get("engagement", 0)) / 3, 1)
            rows.append({
                "Student":          s.get("name", "Unknown"),
                "Relevance":        sc.get("relevance", "–"),
                "Knowledgeability": sc.get("knowledgeability", "–"),
                "Engagement":       sc.get("engagement", "–"),
                "Average":          avg,
            })
        rows.sort(key=lambda r: r["Average"], reverse=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
#  TAB 2 — Student Dashboard
# ════════════════════════════════════════════

with tab_student:

    if not student_ev:
        st.warning("No student evaluation data was returned by the model.")
        st.stop()

    student_names = [s.get("name", f"Student {i}") for i, s in enumerate(student_ev)]

    selected_name = st.selectbox(
        "Select a Student",
        options=student_names,
        help="Choose a student to view their detailed participation report.",
    )

    student = next((s for s in student_ev if s.get("name") == selected_name), None)

    if student is None:
        st.error("Could not load data for the selected student.")
        st.stop()

    scores   = student.get("scores",   {})
    feedback = student.get("feedback", {})

    rel = scores.get("relevance",        0)
    kno = scores.get("knowledgeability", 0)
    eng = scores.get("engagement",       0)
    avg = round((rel + kno + eng) / 3, 1)

    st.markdown(f"### {selected_name}")
    st.caption(f"Overall average score: **{avg} / 10**")

    # ── Score metrics ──
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.metric("📌 Relevance",        f"{rel} / 10",
                  delta=f"{rel-5:+d} vs midpoint" if rel != 5 else None)
    with c2:
        st.metric("🧠 Knowledgeability", f"{kno} / 10",
                  delta=f"{kno-5:+d} vs midpoint" if kno != 5 else None)
    with c3:
        st.metric("🙋 Engagement",       f"{eng} / 10",
                  delta=f"{eng-5:+d} vs midpoint" if eng != 5 else None)
    with c4:
        st.metric("⭐ Average",           f"{avg} / 10",
                  delta=f"{avg-5:+.1f} vs midpoint" if avg != 5 else None)

    st.markdown("---")

    # ── Radar chart + feedback side-by-side ──
    chart_col, feed_col = st.columns([1, 1], gap="large")

    with chart_col:
        st.markdown('<div class="section-title">🕸️ Performance Radar</div>', unsafe_allow_html=True)
        fig = radar_chart(scores, selected_name)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with feed_col:
        st.markdown('<div class="section-title">📝 Detailed Feedback</div>', unsafe_allow_html=True)

        strengths   = feedback.get("strengths",          "Not available.")
        weaknesses  = feedback.get("weaknesses",         "Not available.")
        improvement = feedback.get("needs_improvement",  "Not available.")

        st.markdown(f"""
        <div class="feedback-row strength">
            <div class="fb-label">✅ Strengths</div>
            <div class="fb-text">{strengths}</div>
        </div>
        <div class="feedback-row weakness">
            <div class="fb-label">⚠️ Weaknesses</div>
            <div class="fb-text">{weaknesses}</div>
        </div>
        <div class="feedback-row">
            <div class="fb-label">🎯 Needs Improvement</div>
            <div class="fb-text">{improvement}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Class comparison ──
    if len(student_ev) > 1:
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Class Comparison</div>', unsafe_allow_html=True)

        all_avgs  = [
            round((s["scores"]["relevance"] +
                   s["scores"]["knowledgeability"] +
                   s["scores"]["engagement"]) / 3, 1)
            for s in student_ev
        ]
        class_avg   = round(sum(all_avgs) / len(all_avgs), 1)
        delta_class = round(avg - class_avg, 1)

        co1, co2 = st.columns(2)
        co1.metric("Class Average Score", f"{class_avg} / 10")
        co2.metric(f"{selected_name} vs Class", f"{avg} / 10",
                   delta=f"{delta_class:+.1f}", delta_color="normal")

        bar_names  = [s.get("name", "?") for s in student_ev]
        bar_colors = [
            "#38d9f5" if n == selected_name else "#1e3a6e"
            for n in bar_names
        ]

        bar_fig = go.Figure(go.Bar(
            x=bar_names,
            y=all_avgs,
            marker_color=bar_colors,
            marker_line_color="#2a52a0",
            marker_line_width=1.5,
            text=[f"{v}" for v in all_avgs],
            textposition="outside",
            textfont=dict(color="#ffffff", size=12),
            hovertemplate="%{x}: <b>%{y}/10</b><extra></extra>",
        ))
        bar_fig.update_layout(
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0a1628",
            font=dict(color="#ffffff"),
            yaxis=dict(
                range=[0, 11],
                gridcolor="#1e3a6e",
                zeroline=False,
                tickfont=dict(color="#a8c4e0"),
            ),
            xaxis=dict(tickfont=dict(size=11, color="#ffffff")),
            margin=dict(l=20, r=20, t=30, b=20),
            height=280,
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})
