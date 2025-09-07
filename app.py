# app.py — Streamlit UPSC Mock Test (UI-only, OpenAI via st.secrets)

import time
import json
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st

# Optional OpenAI client (used only if provider == "OpenAI")
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ======================= App Config =======================
st.set_page_config(page_title="UPSC Mock Test", layout="wide")

# ======================= Rate Limiting =======================
COOLDOWN_SECONDS = 30
DAILY_LIMIT = 40
HOURLY_SHARED_CAP = 250  # set 0 to disable shared-hourly cap
# === Cost guardrail ===
DAILY_BUDGET = 1.00        # hard cap in dollars per day
EST_COST_PER_GEN = 1.00    # rough cost per generation (use 1.00 to be safe)


def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")

@st.cache_resource
def _shared_hourly_counters():
    return {}

def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    if "rl_last_ts" not in ss:
        ss["rl_last_ts"] = 0.0
    if "rl_calls_today" not in ss:
        ss["rl_calls_today"] = 0

def can_call_now():
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()

    # Cooldown
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Please wait {remaining}s before the next generation.", remaining)

    # Daily budget check (primary guardrail)
    est_spend = ss["rl_calls_today"] * EST_COST_PER_GEN
    if est_spend >= DAILY_BUDGET:
        return (False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}). Try again tomorrow.", 0)

    # Optional: also keep your per-session daily cap (can leave as-is or lower)
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations). Try again tomorrow.", 0)

    # Optional shared hourly cap
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        used = counters.get(bucket, 0)
        if used >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached. Please try later.", 0)

    return (True, "", 0)

def record_successful_call():
    ss = st.session_state
    ss["rl_last_ts"] = time.time()
    ss["rl_calls_today"] += 1
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        counters[bucket] = counters.get(bucket, 0) + 1

# ======================= Data Models =======================
@dataclass
class MCQ:
    id: str
    question: str
    options: List[str]
    correct_index: int
    explanation: str

@dataclass
class Essay:
    id: str
    prompt: str
    rubric_points: List[str]

# ======================= UI Helpers =======================
def brand_h2(text: str, color: str):
    st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{color}'>{text}</h2>", unsafe_allow_html=True)

def md_card(title_text: str, body_html: str = ""):
    extra = f'<div style="margin-top:.35rem">{body_html}</div>' if body_html else ""
    st.markdown(
        f"""
<div style="border:1px solid #e5e7eb; padding:.75rem 1rem; border-radius:10px; margin-bottom:.75rem;">
  <div style="font-weight:600">{title_text}</div>
  {extra}
</div>
        """,
        unsafe_allow_html=True
    )

# ======================= Topic Catalogs =======================
GS_PRELIMS = [
    "Polity & Governance", "Economy", "Geography", "History & Culture",
    "Environment & Ecology", "Science & Tech", "Current Affairs"
]
CSAT_SECTIONS = ["Comprehension", "Reasoning", "Data Interpretation", "Basic Numeracy"]
MAINS_GS = [
    "GS1: History & Culture, Society, Geography",
    "GS2: Polity, Governance, IR",
    "GS3: Economy, Agriculture, S&T, Environment, Security",
    "GS4: Ethics, Integrity, Aptitude",
    "Essay"
]
OPTIONALS = [
    "Public Administration", "Sociology", "Anthropology", "Geography (Optional)",
    "History (Optional)", "Political Science & IR", "Economics (Optional)",
    "Psychology", "Philosophy", "Mathematics", "Management"
]

# ======================= Offline Generators =======================
def offline_mcq_bank(topic: str, difficulty: str, language: str, seed: int, n: int) -> List[MCQ]:
    rng = random.Random(seed + len(topic) + len(difficulty) + len(language))
    mcqs: List[MCQ] = []
    stems = [
        "Which of the following statements is/are correct regarding {X}?",
        "Consider the following statements about {X}. Which of the statements given above is/are correct?",
        "With reference to {X}, consider the following: choose the correct option."
    ]
    facts = [
        "{X} is constitutionally backed.",
        "{X} affects federal-state relations.",
        "{X} influences inclusive growth.",
        "{X} has implications for climate resilience.",
        "{X} is linked to demographic trends.",
        "{X} is notified under a recent policy."
    ]
    exps = [
        "Statement 1 is correct because of its legal basis; Statement 2 is incorrect due to scope limits.",
        "The provision applies conditionally; hence only one statement holds.",
        "Recent committee reports clarify the scope, aligning with option chosen."
    ]
    for i in range(n):
        stem = rng.choice(stems).format(X=topic)
        s1 = rng.choice(facts).format(X=topic)
        s2 = rng.choice(facts).format(X=topic)
        options = ["1 only", "2 only", "Both 1 and 2", "Neither 1 nor 2"]
        rng.shuffle(options)
        truth = rng.choice(options)
        correct_index = options.index(truth)
        qtxt = f"{stem}\n\n1. {s1}\n2. {s2}\n"
        exp = rng.choice(exps)
        mcqs.append(MCQ(
            id=f"{topic}-{i}-{rng.randint(1000,9999)}",
            question=qtxt,
            options=options,
            correct_index=correct_index,
            explanation=exp
        ))
    return mcqs

def offline_essays(topic: str, difficulty: str, language: str, seed: int, n: int) -> List[Essay]:
    rng = random.Random(seed + len(topic) + len(difficulty) + len(language) + 99)
    prompts = [
        "Critically examine the role of {X} in advancing inclusive development.",
        "Discuss how {X} reshapes federal dynamics and governance outcomes.",
        "Evaluate the challenges and opportunities of {X} for sustainable growth.",
        "Analyze ethical considerations surrounding {X} in public administration.",
        "How does {X} interact with technological change and social equity?"
    ]
    rubrics = [
        ["Concept clarity", "Use of examples/data", "Critical analysis", "Structure & coherence", "Conclusion"],
        ["Understanding of syllabus demand", "Interlinkages across topics", "Counter-arguments", "Recommendations", "Language"]
    ]
    essays: List[Essay] = []
    for i in range(n):
        prompt = rng.choice(prompts).format(X=topic)
        rubric = rng.choice(rubrics)
        essays.append(Essay(
            id=f"essay-{topic}-{i}-{rng.randint(1000,9999)}",
            prompt=prompt,
            rubric_points=rubric
        ))
    return essays

# ======================= OpenAI Generators (JSON-only) =======================
def call_openai_mcq(model: str, topic: str, difficulty: str, language: str, n: int, temperature: float, max_tokens: int) -> List[MCQ]:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Add openai to requirements.txt.")
    client = OpenAI(api_key=api_key)

    sys = (
        "You are a UPSC Prelims question setter. Output strict JSON only with key 'mcqs' as a list of objects. "
        "Each object must have: id (string), question (string, can include two statements labeled 1 and 2), "
        "options (array of 4 strings), correct_index (0-3), explanation (string). No extra keys or prose."
    )
    usr = json.dumps({
        "topic": topic,
        "difficulty": difficulty,
        "language": language,
        "count": n,
        "style": "two-statement style preferred; UPSC tone; balanced difficulty; avoid niche facts."
    })

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    data = json.loads(text)
    out: List[MCQ] = []
    for q in data.get("mcqs", []):
        out.append(MCQ(
            id=str(q.get("id", "")),
            question=str(q.get("question", "")),
            options=[str(x) for x in q.get("options", [])][:4],
            correct_index=int(q.get("correct_index", 0)),
            explanation=str(q.get("explanation", "")),
        ))
    return out

def call_openai_essay(model: str, topic: str, difficulty: str, language: str, n: int, temperature: float, max_tokens: int) -> List[Essay]:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Add openai to requirements.txt.")
    client = OpenAI(api_key=api_key)

    sys = (
        "You are a UPSC Mains question setter. Output strict JSON only with key 'essays' as a list of objects. "
        "Each object must have: id (string), prompt (string), rubric_points (array of short strings). No extra keys or prose."
    )
    usr = json.dumps({
        "topic": topic,
        "difficulty": difficulty,
        "language": language,
        "count": n,
        "style": "UPSC Mains tone; analytical; allow multidimensional treatment; avoid niche trivia."
    })

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    data = json.loads(text)
    out: List[Essay] = []
    for e in data.get("essays", []):
        out.append(Essay(
            id=str(e.get("id", "")),
            prompt=str(e.get("prompt", "")),
            rubric_points=[str(x) for x in e.get("rubric_points", [])]
        ))
    return out

# ======================= Sidebar (Common Controls) =======================
st.title("UPSC Mock Test")

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = st.color_picker("Brand color", value="#0F62FE")
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 512, 4096, 1500, 32)


    init_rate_limit_state()
    ss = st.session_state

    st.markdown("**Usage limits**")
    st.markdown(f"<span style='font-size:0.9rem'>Today: {ss['rl_calls_today']} / {DAILY_LIMIT} generations</span>", unsafe_allow_html=True)
    
    if HOURLY_SHARED_CAP > 0:
        counters = _shared_hourly_counters()
        used = counters.get(_hour_bucket(), 0)
        st.markdown(f"<span style='font-size:0.9rem'>Hour capacity: {used} / {HOURLY_SHARED_CAP}</span>", unsafe_allow_html=True)
    
        est_spend = ss['rl_calls_today'] * EST_COST_PER_GEN
        st.markdown(
            f"<span style='font-size:0.9rem'>Budget: &#36;{est_spend:.2f} / &#36;{DAILY_BUDGET:.2f}</span>",
            unsafe_allow_html=True
        )

    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - time.time()))
    if remaining > 0:
        st.progress(min(1.0, (COOLDOWN_SECONDS - remaining) / COOLDOWN_SECONDS))
        st.caption(f"Cooldown: {remaining}s")

# ======================= Test Configuration =======================
colA, colB = st.columns([1.3, 1])
with colA:
    exam_type = st.selectbox("Exam Type", ["Prelims (MCQ)", "Mains (Descriptive)"])
    language = st.selectbox("Language", ["English", "Hindi"])
    difficulty = st.selectbox("Difficulty", ["Any", "Easy", "Moderate", "Hard"])
    time_limit = st.number_input("Time limit (minutes)", min_value=0, max_value=300, value=30, step=5)
with colB:
    if exam_type.startswith("Prelims"):
        prelims_topics = st.multiselect("Prelims Topics (GS, CSAT, Current Affairs)", GS_PRELIMS + CSAT_SECTIONS, default=["Polity & Governance", "Economy"])
        num_questions = st.slider("Number of questions", 5, 100, 20, 5)
        negative_mark = st.select_slider("Negative marking", options=[0.0, -0.25, -0.33, -0.5], value=-0.33)
        shuffle_q = st.checkbox("Shuffle questions", value=True)
        # Give the checkbox a key so we can read it later
        show_explanations_after = st.checkbox("Show explanations after submit", value=True, key="show_explanations_after")
    else:
        mains_topics = st.multiselect("Mains Topics (GS, Essay)", MAINS_GS, default=["GS2: Polity, Governance, IR"])
        optional_subject = st.selectbox("Optional Subject (optional)", ["None"] + OPTIONLS if False else ["None"] + OPTIONALS, index=0)
        essay_count = st.slider("Number of questions/prompts", 1, 10, 4, 1)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
allowed, reason, _wait = can_call_now()
with col1:
    gen = st.button("Generate Test", type="primary", disabled=not allowed)
with col2:
    regen = st.button("Regenerate")
with col3:
    start_timer = st.button("Start Timer")
with col4:
    submit = st.button("Submit")

reset = st.button("Reset All")

# ======================= Session State =======================
if "seed" not in st.session_state:
    st.session_state.seed = 101
if "test_started_at" not in st.session_state:
    st.session_state.test_started_at = None
if "prelims_qs" not in st.session_state:
    st.session_state.prelims_qs: List[MCQ] = []
if "mains_qs" not in st.session_state:
    st.session_state.mains_qs: List[Essay] = []
if "answers" not in st.session_state:
    st.session_state.answers: Dict[str, int] = {}
if "essay_answers" not in st.session_state:
    st.session_state.essay_answers: Dict[str, str] = {}

if reset:
    for k in ["prelims_qs", "mains_qs", "answers", "essay_answers", "test_started_at"]:
        st.session_state.pop(k, None)

# ======================= Generation Orchestrators =======================
def generate_prelims(topics: List[str], n: int):
    blocks: List[MCQ] = []
    if not topics:
        return blocks
    per_topic = max(1, n // len(topics))
    remainder = n - per_topic * len(topics)
    for i, t in enumerate(topics):
        want = per_topic + (1 if i < remainder else 0)
        if provider == "Offline (rule-based)":
            blocks.extend(offline_mcq_bank(t, difficulty, language, st.session_state.seed + i * 13, want))
        else:
            try:
                blocks.extend(call_openai_mcq(model, t, difficulty, language, want, temp, max_tokens))
            except Exception as e:
                st.error(f"OpenAI MCQ error for {t}: {e}. Falling back offline for this topic.")
                blocks.extend(offline_mcq_bank(t, difficulty, language, st.session_state.seed + i * 13, want))
    if shuffle_q:
        random.Random(st.session_state.seed).shuffle(blocks)
    return blocks[:n]

def generate_mains(topics: List[str], opt_subject: Optional[str], n: int):
    prompts: List[Essay] = []
    pools = topics.copy()
    if opt_subject and opt_subject != "None":
        pools.append(f"Optional: {opt_subject}")
    for i, t in enumerate(pools):
        if provider == "Offline (rule-based)":
            prompts.extend(offline_essays(t, difficulty, language, st.session_state.seed + i * 29, 1))
        else:
            try:
                prompts.extend(call_openai_essay(model, t, difficulty, language, 1, temp, max_tokens))
            except Exception as e:
                st.error(f"OpenAI Essay error for {t}: {e}. Falling back offline for this topic.")
                prompts.extend(offline_essays(t, difficulty, language, st.session_state.seed + i * 29, 1))
        if len(prompts) >= n:
            break
    return prompts[:n]

# ======================= Actions =======================
if (gen or regen):
    if not allowed:
        st.warning(reason)
    else:
        if exam_type.startswith("Prelims"):
            st.session_state.prelims_qs = generate_prelims(prelims_topics, num_questions)
        else:
            st.session_state.mains_qs = generate_mains(mains_topics, optional_subject, essay_count)
        st.session_state.answers = {}
        st.session_state.essay_answers = {}
        record_successful_call()
        if regen:
            st.session_state.seed += 7

if start_timer and time_limit > 0:
    st.session_state.test_started_at = time.time()

# ======================= Timer =======================
def render_timer():
    if st.session_state.test_started_at and time_limit > 0:
        elapsed = int(time.time() - st.session_state.test_started_at)
        remaining = max(0, time_limit * 60 - elapsed)
        mins = remaining // 60
        secs = remaining % 60
        st.info(f"Time remaining: {mins:02d}:{secs:02d}")
        if remaining == 0:
            st.warning("Time is up. You can still submit to view results.")
    else:
        st.caption("Timer not started.")

# ======================= Render: Prelims =======================
def render_prelims():
    brand_h2("Prelims — MCQ", brand)
    render_timer()

    if not st.session_state.prelims_qs:
        st.info("Click Generate Test to create questions.")
        return

    for i, q in enumerate(st.session_state.prelims_qs, start=1):
        st.markdown(f"**Q{i}.** {q.question}")
        widget_key = f"ans_{q.id}"             # unique across runs/shuffles
        current = st.session_state.answers.get(q.id, None)
        choice = st.radio(
            "Select an option:",
            options=list(range(len(q.options))),
            format_func=lambda idx: f"{chr(65+idx)}. {q.options[idx]}",
            index=current if current is not None else 0,
            key=widget_key
        )
        st.session_state.answers[q.id] = choice
        st.markdown("---")

    if submit:
        total = len(st.session_state.prelims_qs)
        correct = 0
        wrong = 0
        unattempted = 0
        for q in st.session_state.prelims_qs:
            sel = st.session_state.answers.get(q.id, None)
            if sel is None:
                unattempted += 1
            elif sel == q.correct_index:
                correct += 1
            else:
                wrong += 1
        score = correct * 1.0 + wrong * negative_mark

        summary_html = (
            f"- Questions: <b>{total}</b><br>"
            f"- Correct: <b>{correct}</b><br>"
            f"- Wrong: <b>{wrong}</b><br>"
            f"- Unattempted: <b>{unattempted}</b><br>"
            f"- Negative marking: <b>{negative_mark} per wrong</b><br>"
            f"- <b>Score: {score:.2f} / {total:.2f}</b>"
        )
        md_card("Result Summary", body_html=summary_html)

        show_flag = st.session_state.get("show_explanations_after", True)
        if show_flag:
            brand_h2("Review & Explanations", brand)
            for i, q in enumerate(st.session_state.prelims_qs, start=1):
                sel = st.session_state.answers.get(q.id, None)
                correct_tag = chr(65 + q.correct_index)
                your_tag = "-" if sel is None else chr(65 + sel)
                body_html = (
                    f"Correct: <b>{correct_tag}</b> | Your answer: <b>{your_tag}</b><br><br>"
                    f"<b>Explanation:</b> {q.explanation}"
                )
                md_card(f"Q{i}.", body_html=body_html)

# ======================= Render: Mains =======================
def render_mains():
    brand_h2("Mains — Descriptive", brand)
    render_timer()

    if not st.session_state.mains_qs:
        st.info("Click Generate Test to create prompts.")
        return

    for i, e in enumerate(st.session_state.mains_qs, start=1):
        st.markdown(f"**Q{i}.** {e.prompt}")
        st.caption("Rubric points: " + " · ".join(e.rubric_points))
        key = f"essay_{e.id}"
        val = st.session_state.essay_answers.get(e.id, "")
        ans = st.text_area("Your answer:", value=val, height=200, key=key)
        st.session_state.essay_answers[e.id] = ans
        st.markdown("---")

    if submit:
        attempted = sum(1 for v in st.session_state.essay_answers.values() if v.strip())
        total = len(st.session_state.mains_qs)
        md_card(
            "Submission Saved",
            body_html=(
                f"- Prompts: <b>{total}</b><br>"
                f"- Attempted: <b>{attempted}</b><br><br>"
                "Note: Mains answers are not auto-graded in this UI. Use rubric points for self-evaluation."
            )
        )

    with st.expander("Copy this test as Markdown"):
        md_lines = []
        for i, e in enumerate(st.session_state.mains_qs, start=1):
            md_lines.append(f"**Q{i}.** {e.prompt}")
            md_lines.append("Rubric: " + ", ".join(e.rubric_points))
            md_lines.append("")
        st.code("\n".join(md_lines), language="markdown")

# ======================= Main Render =======================
if exam_type.startswith("Prelims"):
    render_prelims()
else:
    render_mains()
