"""
Class Participation Evaluation Dashboard
=========================================
A Streamlit app that parses Zoom/Google Meet transcripts (.vtt or .txt),
sends them to the Groq API (LLaMA-3.3-70b-versatile), and renders
interactive dashboards for professors and students.

5 Evaluation Factors (research-backed):
  1. Relevance          — Are answers on-topic and pertinent?
  2. Knowledgeability   — Depth of subject understanding shown
  3. Engagement         — Active participation frequency & enthusiasm
  4. Critical Thinking  — Ability to analyse, question, and reason
  5. Communication      — Clarity, articulation, and listening

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
#  Custom CSS — blue & white theme (visibility fixed)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:       #0a1628;
    --bg-mid:        #0d1e3d;
    --bg-card:       #102048;
    --accent-cyan:   #38d9f5;
    --accent-yellow: #ffd166;
    --accent-green:  #06d6a0;
    --accent-red:    #ff6b6b;
    --accent-purple: #b388ff;
    --text-white:    #ffffff;
    --text-light:    #a8c4e0;
    --border:        #1e3a6e;
    --border-bright: #2a52a0;
    --radius:        12px;
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
html, body, p, li, div { font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3, h4 { font-family: 'DM Serif Display', serif !important; color: #ffffff !important; }

/* ════════════════════════════════
   SIDEBAR — dark navy
   ════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061022 0%, #0a1830 100%) !important;
    border-right: 2px solid var(--border-bright) !important;
}

/* Sidebar heading */
[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent-cyan) !important;
    font-size: 1.05rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: .4rem;
}

/* ── FIX 1: Professor name label — black text on white/light input ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] label,
[data-testid="stSidebar"] [data-testid="stTextInput"] p {
    color: #ffffff !important;
    font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: #ffffff !important;
    color: #111111 !important;
    border: 2px solid var(--border-bright) !important;
    border-radius: var(--radius) !important;
    font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: #666666 !important;
}

/* ── FIX 2: File uploader — black font on white background area ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed var(--border-bright) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #111111 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #111111 !important;
}
/* File uploader section label above the box */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label,
[data-testid="stSidebar"] section[data-testid="stFileUploader"] > div > label {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #111111 !important;
}

/* ── FIX 3: Student selectbox — dark background with white text ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div {
    background: #102048 !important;
    border: 2px solid var(--border-bright) !important;
    border-radius: var(--radius) !important;
    color: #ffffff !important;
}
[data-testid="stSelectbox"] > div > div > div > div,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] p {
    color: #ffffff !important;
}
[data-testid="stSelectbox"] label {
    color: var(--text-light) !important;
    font-weight: 500;
}
[data-testid="stSelectbox"] svg { fill: var(--accent-cyan) !important; }
/* Dropdown list popup */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] {
    background: #102048 !important;
    color: #ffffff !important;
    border: 1px solid var(--border-bright) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="option"]:hover {
    background: #1e3a6e !important;
    color: #ffffff !important;
}
[data-baseweb="select"] * { color: #ffffff !important; }

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
[data-testid="stMetricLabel"] { color: var(--text-light) !important; font-size:.8rem !important; }
[data-testid="stMetricValue"] { color: var(--accent-cyan) !important; font-size:1.9rem !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { color: var(--accent-green) !important; }

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; }
[data-testid="stAlert"] { background: rgba(16,32,72,0.8) !important; border-color: var(--border-bright) !important; }
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
[data-testid="stExpander"] summary,
[data-testid="stExpander"] * { color: #ffffff !important; }

/* ── Sidebar button ── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: 1.5px solid #42a5f5 !important;
    border-radius: var(--radius) !important;
    padding: .6rem 1.8rem !important;
    font-size: .95rem !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 6px 24px rgba(56,217,245,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid var(--border-bright); }
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text-light) !important;
    padding: .5rem 1.2rem .7rem !important;
    font-size: 1rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all .2s;
}
[data-testid="stTabs"] button[role="tab"]:hover { background: rgba(56,217,245,0.08) !important; color: #ffffff !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 3px solid var(--accent-cyan) !important;
    background: rgba(56,217,245,0.06) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1.5px solid var(--border-bright) !important; border-radius: var(--radius) !important; }
[data-testid="stDataFrame"] * { color: #ffffff !important; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0d2060 0%, #0a1628 70%);
    border: 1.5px solid var(--border-bright);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex; align-items: center; gap: 1.5rem;
    box-shadow: 0 8px 40px rgba(56,217,245,0.1);
    position: relative; overflow: hidden;
}
.hero-header::before {
    content:''; position:absolute; top:-50px; right:-50px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(56,217,245,0.14), transparent 70%);
    border-radius:50%;
}
.hero-header .icon { font-size:3.2rem; filter:drop-shadow(0 0 14px rgba(56,217,245,0.55)); z-index:1; }
.hero-header h1   { margin:0; font-size:2rem; color:#ffffff !important; z-index:1; }
.hero-header p    { margin:.35rem 0 0; color:var(--text-light); font-size:.95rem; z-index:1; }

/* ── Insight card ── */
.insight-card {
    background: linear-gradient(135deg, #102048, #0d1e3d);
    border: 1.5px solid var(--border-bright);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: box-shadow .2s, border-color .2s;
}
.insight-card:hover { border-color:var(--accent-cyan); box-shadow:0 6px 28px rgba(56,217,245,0.14); }
.insight-card .label { font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:var(--accent-cyan); margin-bottom:.5rem; font-weight:600; }
.insight-card .value { color:#ffffff; line-height:1.7; font-size:.95rem; }

/* ── Section title ── */
.section-title {
    color:#ffffff !important;
    border-left:4px solid var(--accent-cyan);
    padding-left:.75rem;
    margin:1.5rem 0 .8rem;
    font-family:'DM Serif Display', serif;
    font-size:1.15rem;
}

/* ── Tag pill ── */
.tag {
    display:inline-block;
    background:rgba(255,209,102,0.15);
    color:var(--accent-yellow);
    border:1px solid rgba(255,209,102,0.35);
    border-radius:20px;
    padding:.25rem .85rem;
    font-size:.82rem; margin:.2rem .18rem; font-weight:500;
}

/* ── Student chips ── */
.student-chip {
    display:inline-block;
    background:rgba(6,214,160,0.12);
    color:var(--accent-green);
    border:1px solid rgba(6,214,160,0.3);
    border-radius:20px;
    padding:.22rem .8rem;
    font-size:.83rem; margin:.15rem; font-weight:500;
}

/* ── Feedback rows ── */
.feedback-row {
    background:linear-gradient(135deg,#102048,#0d1e3d);
    border-left:4px solid var(--accent-yellow);
    border-radius:0 var(--radius) var(--radius) 0;
    padding:1rem 1.2rem; margin-bottom:.8rem;
    box-shadow:0 3px 14px rgba(0,0,0,0.2);
}
.feedback-row.strength { border-left-color:var(--accent-green); }
.feedback-row.weakness { border-left-color:var(--accent-red); }
.feedback-row .fb-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:var(--text-light); margin-bottom:.35rem; font-weight:600; }
.feedback-row .fb-text  { color:#ffffff; line-height:1.7; font-size:.93rem; }

/* ── Score badge ── */
.score-badge {
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(56,217,245,0.1);
    border:1px solid rgba(56,217,245,0.25);
    border-radius:8px; padding:.3rem .7rem;
    font-size:.82rem; color:#ffffff; margin:.15rem;
}
.score-badge .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }

hr { border-color:var(--border-bright) !important; opacity:.5; }
[data-testid="stCaptionContainer"], small, .stCaption { color:var(--text-light) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def get_groq_client() -> Groq:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("⚠️ Groq API key not found. Add `GROQ_API_KEY` to `.streamlit/secrets.toml`.")
        st.stop()
    return Groq(api_key=api_key)


def parse_transcript(raw_text: str) -> str:
    text = re.sub(r"^WEBVTT.*?\n\n", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── Research-backed 5-factor schema ───
# Sources:
#  • Relevance & Knowledgeability — classic participation rubrics (Bean & Peterson, 1998)
#  • Engagement — WICAT model (Márquez et al., 2023, SAGE)
#  • Critical Thinking — ScienceDirect class participation competencies (2023)
#  • Communication Clarity — UNSW Teaching Gateway rubric; active listening dimension

SYSTEM_PROMPT_TEMPLATE = """
You are an expert educational analyst specialising in higher-education participation assessment.
You will receive a transcript from an online class session.

The professor in this session is: {professor_name}

Evaluate every student (anyone who spoke other than the professor) across FIVE research-backed factors:

1. relevance          (1-10) — How on-topic and pertinent were the student's contributions?
2. knowledgeability   (1-10) — How deeply did the student demonstrate subject understanding?
3. engagement         (1-10) — How actively, consistently, and enthusiastically did the student participate?
4. critical_thinking  (1-10) — Did the student analyse, question, reason, or build on ideas critically?
5. communication      (1-10) — How clearly, articulately, and confidently did the student express themselves?

Return ONLY a valid JSON object (no markdown fences, no explanation) with EXACTLY this structure:

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
        "relevance":         <integer 1-10>,
        "knowledgeability":  <integer 1-10>,
        "engagement":        <integer 1-10>,
        "critical_thinking": <integer 1-10>,
        "communication":     <integer 1-10>
      }},
      "feedback": {{
        "strengths":         "<What the student did well across the five dimensions>",
        "weaknesses":        "<Where the student fell short>",
        "needs_improvement": "<One or two specific, actionable improvement steps>"
      }}
    }}
  ]
}}

Rules:
- Only include students who actually spoke (exclude the professor).
- All scores must be integers 1–10.
- topics_to_review: concepts students seemed confused about.
- Return ONLY the JSON — no markdown, no preamble, no explanation.
"""


def call_groq(transcript: str, professor_name: str) -> dict:
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
        with st.expander("Raw LLM output (debug)"):
            st.code(raw, language="json")
        st.stop()


# ─── Factor metadata ───
FACTORS = [
    {"key": "relevance",         "label": "Relevance",         "icon": "📌", "color": "#38d9f5"},
    {"key": "knowledgeability",  "label": "Knowledgeability",  "icon": "🧠", "color": "#ffd166"},
    {"key": "engagement",        "label": "Engagement",        "icon": "🙋", "color": "#06d6a0"},
    {"key": "critical_thinking", "label": "Critical Thinking", "icon": "💡", "color": "#b388ff"},
    {"key": "communication",     "label": "Communication",     "icon": "🗣️", "color": "#ff9f68"},
]


def radar_chart(scores: dict, name: str) -> go.Figure:
    cats   = [f["label"] for f in FACTORS]
    vals   = [scores.get(f["key"], 0) for f in FACTORS]
    colors = [f["color"] for f in FACTORS]

    # close polygon
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c,
        fill="toself",
        fillcolor="rgba(56,217,245,0.15)",
        line=dict(color="#38d9f5", width=2.5),
        name=name,
        hovertemplate="%{theta}: <b>%{r}/10</b><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1e3d",
            radialaxis=dict(
                visible=True, range=[0, 10],
                tickfont=dict(color="#a8c4e0", size=9),
                gridcolor="#1e3a6e", linecolor="#1e3a6e",
            ),
            angularaxis=dict(
                tickfont=dict(color="#ffffff", size=11, family="DM Serif Display"),
                gridcolor="#1e3a6e", linecolor="#1e3a6e",
            ),
        ),
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0a1628",
        font=dict(color="#ffffff"),
        margin=dict(l=50, r=50, t=60, b=50),
        showlegend=False,
        height=380,
    )
    return fig


def bar_comparison_chart(student_ev: list, selected_name: str) -> go.Figure:
    names = [s.get("name", "?") for s in student_ev]
    avgs  = [
        round(sum(s["scores"].get(f["key"], 0) for f in FACTORS) / len(FACTORS), 1)
        for s in student_ev
    ]
    colors = ["#38d9f5" if n == selected_name else "#1e3a6e" for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=avgs,
        marker_color=colors,
        marker_line_color="#2a52a0", marker_line_width=1.5,
        text=avgs, textposition="outside",
        textfont=dict(color="#ffffff", size=11),
        hovertemplate="%{x}: <b>%{y}/10</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        font=dict(color="#ffffff"),
        yaxis=dict(range=[0,11], gridcolor="#1e3a6e", zeroline=False, tickfont=dict(color="#a8c4e0")),
        xaxis=dict(tickfont=dict(size=11, color="#ffffff")),
        margin=dict(l=20,r=20,t=30,b=20),
        height=280,
    )
    return fig


def factor_breakdown_chart(scores: dict) -> go.Figure:
    """Horizontal bar chart showing all 5 factor scores."""
    labels = [f["label"] for f in FACTORS]
    values = [scores.get(f["key"], 0) for f in FACTORS]
    colors = [f["color"] for f in FACTORS]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=1,
        text=[f"{v}/10" for v in values],
        textposition="outside",
        textfont=dict(color="#ffffff", size=12),
        hovertemplate="%{y}: <b>%{x}/10</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a1628", plot_bgcolor="#0d1e3d",
        font=dict(color="#ffffff"),
        xaxis=dict(range=[0,12], gridcolor="#1e3a6e", zeroline=False, tickfont=dict(color="#a8c4e0")),
        yaxis=dict(tickfont=dict(color="#ffffff", size=12)),
        margin=dict(l=10, r=60, t=20, b=20),
        height=240,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Setup")
    st.markdown("---")

    professor_name = st.text_input(
        "Professor's Name",
        placeholder="e.g. Dr. Sarah Chen",
        help="Used to distinguish the professor from students in the transcript.",
    )

    uploaded_file = st.file_uploader(
        "Upload Transcript",
        type=["vtt", "txt"],
        help="Supports Zoom / Google Meet .vtt exports or plain-text with Speaker: dialogue format.",
    )

    analyse_btn = st.button("⚡ Analyse Transcript", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#a8c4e0'>Powered by Groq · LLaMA 3.3 70B<br>"
        "5-Factor Evaluation · Built with Streamlit</small>",
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
        <p>AI-powered · 5-Factor Assessment · Research-backed Evaluation Framework</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────

if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ─────────────────────────────────────────────
#  Trigger
# ─────────────────────────────────────────────

if analyse_btn:
    if uploaded_file is None:
        st.warning("Please upload a transcript file first.")
    elif not professor_name.strip():
        st.warning("Please enter the professor's name.")
    else:
        raw_text  = uploaded_file.read().decode("utf-8", errors="replace")
        clean_txt = parse_transcript(raw_text)
        with st.spinner("🔍 Analysing transcript across 5 dimensions — this may take 20–40 seconds…"):
            st.session_state.analysis = call_groq(clean_txt, professor_name.strip())
        st.success("✅ Analysis complete! Explore the dashboards below.")

# ─────────────────────────────────────────────
#  Render
# ─────────────────────────────────────────────

if st.session_state.analysis is None:
    st.info(
        "👈 Upload a transcript and click **Analyse Transcript** to get started.\n\n"
        "**Supported formats:** Zoom `.vtt`, Google Meet `.txt`, or any plain-text with `Speaker: text` format."
    )

    # Factor legend
    st.markdown("---")
    st.markdown("#### 📐 5-Factor Evaluation Framework")
    cols = st.columns(5)
    descs = [
        "On-topic, pertinent answers",
        "Depth of subject knowledge",
        "Active & consistent participation",
        "Analysis, reasoning & questioning",
        "Clarity & articulation",
    ]
    for col, f, desc in zip(cols, FACTORS, descs):
        col.markdown(
            f'<div class="insight-card" style="text-align:center">'
            f'<div style="font-size:1.8rem">{f["icon"]}</div>'
            f'<div style="color:{f["color"]};font-weight:600;font-size:.85rem;margin:.4rem 0">{f["label"]}</div>'
            f'<div style="color:#a8c4e0;font-size:.78rem">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
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

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
        <div class="insight-card">
            <div class="label">📊 Overall Class Understanding</div>
            <div class="value">{prof_dash.get("overall_class_understanding","N/A")}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="insight-card">
            <div class="label">💡 Teaching Feedback</div>
            <div class="value">{prof_dash.get("teaching_feedback","N/A")}</div>
        </div>""", unsafe_allow_html=True)

    # Topics
    topics = prof_dash.get("topics_to_review", [])
    if topics:
        st.markdown('<div class="section-title">🔖 Topics to Review</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="tag">{t}</span>' for t in topics), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Question mapping
    qmapping = prof_dash.get("question_mapping", [])
    if qmapping:
        st.markdown('<div class="section-title">❓ Question Mapping</div>', unsafe_allow_html=True)
        st.caption("Each question asked by the professor, with students who responded.")
        for i, item in enumerate(qmapping, 1):
            with st.expander(f"Q{i}: {item.get('professor_question', f'Question {i}')}"):
                answerers = item.get("students_who_answered", [])
                if answerers:
                    chips = "".join(f'<span class="student-chip">👤 {s}</span>' for s in answerers)
                    st.markdown(
                        f'<span style="color:#a8c4e0;font-size:.8rem">Answered by: </span>{chips}',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<span style="color:#ff6b6b">⚠️ No students answered.</span>', unsafe_allow_html=True)

    # 5-factor class summary table
    if student_ev:
        st.markdown('<div class="section-title">📈 Full Participation Summary</div>', unsafe_allow_html=True)
        rows = []
        for s in student_ev:
            sc  = s.get("scores", {})
            avg = round(sum(sc.get(f["key"], 0) for f in FACTORS) / len(FACTORS), 1)
            row = {"Student": s.get("name", "Unknown")}
            for f in FACTORS:
                row[f["label"]] = sc.get(f["key"], "–")
            row["Average ⭐"] = avg
            rows.append(row)
        rows.sort(key=lambda r: r["Average ⭐"], reverse=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
#  TAB 2 — Student Dashboard
# ════════════════════════════════════════════

with tab_student:

    if not student_ev:
        st.warning("No student evaluation data returned.")
        st.stop()

    student_names = [s.get("name", f"Student {i}") for i, s in enumerate(student_ev)]

    selected_name = st.selectbox(
        "🎓 Select a Student",
        options=student_names,
        help="Choose a student to view their 5-factor participation report.",
    )

    student  = next((s for s in student_ev if s.get("name") == selected_name), None)
    if not student:
        st.error("Could not load student data.")
        st.stop()

    scores   = student.get("scores",   {})
    feedback = student.get("feedback", {})

    all_scores = [scores.get(f["key"], 0) for f in FACTORS]
    avg        = round(sum(all_scores) / len(FACTORS), 1)

    st.markdown(f"### {selected_name}")
    st.caption(f"5-Factor average: **{avg} / 10**")

    # ── 5 metric cards ──
    cols = st.columns(5, gap="small")
    for col, f in zip(cols, FACTORS):
        val = scores.get(f["key"], 0)
        col.metric(
            f'{f["icon"]} {f["label"]}',
            f"{val} / 10",
            delta=f"{val-5:+d} vs mid" if val != 5 else None,
        )

    st.markdown("---")

    # ── Radar + factor bar side-by-side ──
    radar_col, bar_col = st.columns([1, 1], gap="large")

    with radar_col:
        st.markdown('<div class="section-title">🕸️ Performance Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(scores, selected_name), use_container_width=True, config={"displayModeBar": False})

    with bar_col:
        st.markdown('<div class="section-title">📊 Factor Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(factor_breakdown_chart(scores), use_container_width=True, config={"displayModeBar": False})

        # Score badges
        badge_html = ""
        for f in FACTORS:
            val = scores.get(f["key"], 0)
            badge_html += (
                f'<span class="score-badge">'
                f'<span class="dot" style="background:{f["color"]}"></span>'
                f'{f["icon"]} {f["label"]}: <b>{val}/10</b></span>'
            )
        st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Feedback panel ──
    st.markdown('<div class="section-title">📝 Detailed Feedback</div>', unsafe_allow_html=True)
    fb1, fb2, fb3 = st.columns(3, gap="medium")

    with fb1:
        st.markdown(f"""
        <div class="feedback-row strength">
            <div class="fb-label">✅ Strengths</div>
            <div class="fb-text">{feedback.get("strengths","Not available.")}</div>
        </div>""", unsafe_allow_html=True)

    with fb2:
        st.markdown(f"""
        <div class="feedback-row weakness">
            <div class="fb-label">⚠️ Weaknesses</div>
            <div class="fb-text">{feedback.get("weaknesses","Not available.")}</div>
        </div>""", unsafe_allow_html=True)

    with fb3:
        st.markdown(f"""
        <div class="feedback-row">
            <div class="fb-label">🎯 Needs Improvement</div>
            <div class="fb-text">{feedback.get("needs_improvement","Not available.")}</div>
        </div>""", unsafe_allow_html=True)

    # ── Class comparison ──
    if len(student_ev) > 1:
        st.markdown("---")
        st.markdown('<div class="section-title">🏆 Class Comparison</div>', unsafe_allow_html=True)

        all_avgs  = [
            round(sum(s["scores"].get(f["key"], 0) for f in FACTORS) / len(FACTORS), 1)
            for s in student_ev
        ]
        class_avg   = round(sum(all_avgs) / len(all_avgs), 1)
        delta_class = round(avg - class_avg, 1)

        m1, m2, m3 = st.columns(3)
        m1.metric("Class Average",         f"{class_avg} / 10")
        m2.metric(f"{selected_name}",      f"{avg} / 10",      delta=f"{delta_class:+.1f} vs class")
        m3.metric("Students Evaluated",    str(len(student_ev)))

        st.plotly_chart(
            bar_comparison_chart(student_ev, selected_name),
            use_container_width=True,
            config={"displayModeBar": False},
        )
