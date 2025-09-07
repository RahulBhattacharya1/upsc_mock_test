# UPSC Mock Test (AI + Offline)

An interactive **Streamlit app** that generates **UPSC Prelims mock MCQs** and **Mains descriptive prompts**.

---

## Features

- **Exam Modes**:
  - **Prelims (MCQ)**: UPSC-style two-statement multiple choice questions.
  - **Mains (Descriptive)**: Essay and GS prompts with rubric points.
- **Flexible Topics**:
  - GS Prelims (Polity, Economy, Geography, History, Environment, Science, Current Affairs)
  - CSAT (Comprehension, Reasoning, Data Interpretation, Numeracy)
  - Mains GS (GS1–GS4, Essay)
  - Optional Subjects (e.g. Public Administration, Sociology, Geography, etc.)
- **Controls**:
  - Difficulty: Easy / Moderate / Hard
  - Language: English / Hindi
  - Negative marking (Prelims)
  - Time limit + timer
  - Shuffle questions
  - Show explanations after submit
- **Output**:
  - MCQs with answers + explanations
  - Mains prompts with rubric points
  - Markdown export for all questions
- **Modes of operation**:
  - **Offline (rule-based)**: Generates mock questions without API.
  - **OpenAI mode**: Uses GPT models (gpt-4o, gpt-4.1-mini, etc.) for richer outputs.

---

## Quickstart

1. **Clone or download** this repo.

2. **Install dependencies** (preferably inside a virtual environment):
   ```bash
   pip install -r requirements.txt
