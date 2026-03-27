#!/usr/bin/env python3
"""
Generate publication-quality figures for the Mental Health RAG project.

Reads actual transcript data and output files to produce five visualizations:
1. rag_architecture.png   — Pipeline architecture diagram
2. session_topics.png     — Bar chart of session topics across 12 transcripts
3. risk_assessment.png    — Risk levels across sessions
4. retrieval_scores.png   — Similarity scores from retrieval_example.json
5. api_flow.png           — FastAPI endpoint flow diagram

Usage:
    python scripts/generate_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"

# Color palette
PRIMARY = "#5a9e8f"
ACCENT_1 = "#4f7cac"
ACCENT_2 = "#9b6b9e"
BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
TEXT_COLOR = "#e0e0e0"
TEXT_LIGHT = "#b0b0b0"
GRID_COLOR = "#2a2a4a"
RISK_COLORS = {
    "none": "#5a9e8f",
    "low": "#4f7cac",
    "moderate": "#d4a843",
    "high": "#c0392b",
    "imminent": "#8b0000",
}

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_CARD,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_LIGHT,
    "ytick.color": TEXT_LIGHT,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Data Extraction from Transcripts
# ---------------------------------------------------------------------------

def extract_session_data() -> list[dict]:
    """Parse all transcript files and extract metadata and clinical info."""
    sessions = []
    for filepath in sorted(TRANSCRIPTS_DIR.glob("session_*.txt")):
        content = filepath.read_text(encoding="utf-8")
        text_lower = content.lower()

        # Extract metadata
        session_num = re.search(r"Session Number:\s*(\d+)", content)
        client_id = re.search(r"Client ID:\s*([\w-]+)", content)
        session_type = re.search(r"Session Type:\s*(.+?)(?:\n|$)", content)
        session_date = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", content)

        # Determine primary topic/diagnosis from session type and content
        stype = session_type.group(1).strip() if session_type else ""
        topic = _classify_topic(stype, text_lower)

        # Determine risk level
        risk = _assess_risk_level(text_lower)

        sessions.append({
            "file": filepath.name,
            "session_num": int(session_num.group(1)) if session_num else 0,
            "client_id": client_id.group(1) if client_id else "unknown",
            "session_type": stype,
            "date": session_date.group(1) if session_date else "",
            "topic": topic,
            "risk_level": risk,
            "label": filepath.stem.replace("session_", "S"),
        })

    return sessions


def _classify_topic(session_type: str, text_lower: str) -> str:
    """Classify a session into a primary clinical topic."""
    stype_lower = session_type.lower()

    if "intake" in stype_lower or "depression" in text_lower and "phq-9" in text_lower:
        if "intake" in stype_lower:
            return "Major Depression"
    if "ocd" in stype_lower or "erp" in stype_lower:
        return "OCD (ERP)"
    if "bipolar" in stype_lower:
        return "Bipolar Disorder"
    if "ptsd" in stype_lower or "military" in stype_lower:
        return "PTSD (Military)"
    if "trauma" in stype_lower or "cpt" in stype_lower:
        return "PTSD (Sexual Assault)"
    if "substance" in stype_lower:
        return "Substance Use"
    if "grief" in stype_lower:
        return "Grief/Bereavement"
    if "couple" in stype_lower:
        return "Relationship Distress"
    if "adolescent" in stype_lower:
        return "Adolescent Self-Harm"
    if "adjustment" in stype_lower:
        return "Adjustment Disorder"
    if "cbt" in stype_lower and "maintenance" in stype_lower:
        return "Panic Disorder (Maintenance)"
    if "cbt" in stype_lower:
        return "Panic Disorder"
    if "phq-9" in text_lower:
        return "Major Depression"
    return "General"


def _assess_risk_level(text_lower: str) -> str:
    """Determine risk level from transcript content.

    Carefully distinguishes between risk keywords appearing in therapist
    safety screening questions (which are routine) vs. actual client
    endorsement of risk behaviors.
    """
    # Denial phrases -- client explicitly denies risk
    denial = any(p in text_lower for p in [
        "denies suicidal", "denied suicidal", "no suicidal ideation",
        "denies self-harm", "no self-harm", "no, nothing like that",
        "nothing like that", "no. i'm discouraged but not hopeless",
        "no. i've got too much to live for",
        "i didn't mean i want to kill myself",
        "i wouldn't do anything",
    ])

    # Active risk indicators -- client endorses active ideation
    active = any(p in text_lower for p in [
        "kill myself", "want to die", "plan to harm",
    ])

    # Passive risk indicators -- client endorses passive wishes
    passive = any(p in text_lower for p in [
        "better off without me", "not wanting to be alive",
        "want to disappear", "passive death wishes",
    ])

    # Self-harm with history -- even if in recovery, history of cutting
    # with recent "want to disappear" language elevates risk
    self_harm_with_history = (
        "cutting" in text_lower
        and ("ice cube" in text_lower or "weeks clean" in text_lower
             or "without cutting" in text_lower)
    )

    # Substance-related risk (overdose mentioned as possibility, not plan)
    substance_risk = "overdose" in text_lower and "dead from an overdose" in text_lower

    # Classify
    if self_harm_with_history and passive:
        return "moderate"
    if self_harm_with_history and not denial:
        return "moderate"
    if active and not denial:
        return "moderate"
    if passive and not denial:
        return "low"
    if passive and denial:
        return "low"

    # Check for genuine risk keywords (not just in screening questions)
    # "better off without me" is passive ideation even with denial
    if "better off without me" in text_lower:
        return "low"

    # Substance-related risk
    if substance_risk:
        return "low"

    # If the only mention of risk keywords is in therapist's screening
    # question and client denies, that's no risk
    if denial:
        return "none"

    return "none"


def load_retrieval_data() -> dict:
    """Load retrieval example data from outputs."""
    retrieval_file = OUTPUTS_DIR / "retrieval_example.json"
    if retrieval_file.exists():
        return json.loads(retrieval_file.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Figure 1: RAG Architecture Diagram
# ---------------------------------------------------------------------------

def generate_rag_architecture(save_path: Path) -> None:
    """Generate the RAG pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)

    # Pipeline stages
    stages = [
        ("Therapy\nTranscripts", 1.0, PRIMARY),
        ("Dialogue-Aware\nChunking", 3.2, ACCENT_1),
        ("OpenAI\nEmbeddings", 5.4, ACCENT_2),
        ("ChromaDB\nVector Store", 7.6, PRIMARY),
        ("Similarity\nRetrieval", 9.8, ACCENT_1),
        ("GPT-4o-mini\nSummarizer", 12.0, ACCENT_2),
    ]

    box_w, box_h = 1.8, 1.6
    y_center = 2.5

    for label, x, color in stages:
        box = FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=color + "30",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, y_center, label,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=TEXT_COLOR)

    # Arrows between stages
    arrow_style = "Simple,tail_width=2,head_width=8,head_length=6"
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + box_w / 2
        x2 = stages[i + 1][1] - box_w / 2
        ax.annotate("",
                     xy=(x2, y_center), xytext=(x1, y_center),
                     arrowprops=dict(
                         arrowstyle="->",
                         color=TEXT_LIGHT,
                         lw=1.5,
                         connectionstyle="arc3,rad=0",
                     ))

    # Output box below summarizer
    out_x, out_y = 12.0, 0.6
    out_box = FancyBboxPatch(
        (out_x - 1.0, out_y - 0.4),
        2.0, 0.8,
        boxstyle="round,pad=0.1",
        facecolor="#2a5a4a",
        edgecolor=PRIMARY,
        linewidth=2,
    )
    ax.add_patch(out_box)
    ax.text(out_x, out_y, "Structured\nJSON Output",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=PRIMARY)

    ax.annotate("",
                 xy=(out_x, out_y + 0.4), xytext=(out_x, y_center - box_h / 2),
                 arrowprops=dict(arrowstyle="->", color=TEXT_LIGHT, lw=1.5))

    # Risk assessment branch
    risk_x, risk_y = 9.8, 0.6
    risk_box = FancyBboxPatch(
        (risk_x - 1.0, risk_y - 0.4),
        2.0, 0.8,
        boxstyle="round,pad=0.1",
        facecolor="#5a3a2a",
        edgecolor="#d4a843",
        linewidth=2,
    )
    ax.add_patch(risk_box)
    ax.text(risk_x, risk_y, "Risk\nAssessment",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="#d4a843")

    ax.annotate("",
                 xy=(risk_x, risk_y + 0.4), xytext=(risk_x, y_center - box_h / 2),
                 arrowprops=dict(arrowstyle="->", color="#d4a843", lw=1.2,
                                  linestyle="dashed"))

    # Title
    ax.text(7.0, 4.6, "Mental Health Session Summary RAG — Pipeline Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=TEXT_COLOR)

    # Subtitle annotations
    ax.text(1.0, 0.6, "12 synthetic\nsessions", ha="center", fontsize=8,
            color=TEXT_LIGHT, style="italic")
    ax.text(3.2, 0.6, "1200 char chunks\n200 overlap", ha="center", fontsize=8,
            color=TEXT_LIGHT, style="italic")
    ax.text(5.4, 0.6, "text-embedding\n-3-small", ha="center", fontsize=8,
            color=TEXT_LIGHT, style="italic")
    ax.text(7.6, 0.6, "Cosine similarity\nTop-K=5", ha="center", fontsize=8,
            color=TEXT_LIGHT, style="italic")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Session Topics Bar Chart
# ---------------------------------------------------------------------------

def generate_session_topics(sessions: list[dict], save_path: Path) -> None:
    """Generate a bar chart of session topics/diagnoses."""
    # Count topics
    topic_counts: dict[str, int] = {}
    for s in sessions:
        t = s["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1

    # Sort by count descending
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    topics = [t[0] for t in sorted_topics]
    counts = [t[1] for t in sorted_topics]

    # Color mapping
    palette = [PRIMARY, ACCENT_1, ACCENT_2, "#d4a843", "#c0392b",
               "#5dade2", "#48c9b0", "#f39c12", "#e74c3c", "#8e44ad",
               "#2ecc71", "#e67e22"]
    colors = [palette[i % len(palette)] for i in range(len(topics))]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(range(len(topics)), counts, color=colors, edgecolor="none",
                   height=0.65, alpha=0.85)

    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=11)
    ax.set_xlabel("Number of Sessions", fontsize=12)
    ax.set_title("Clinical Topics Across 12 Therapy Session Transcripts",
                 fontsize=14, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) + 0.8)
    ax.grid(axis="x", alpha=0.2)

    # Value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=11, fontweight="bold",
                color=TEXT_COLOR)

    # Session IDs annotation
    topic_sessions: dict[str, list[str]] = {}
    for s in sessions:
        t = s["topic"]
        if t not in topic_sessions:
            topic_sessions[t] = []
        topic_sessions[t].append(s["label"])

    for i, topic in enumerate(topics):
        session_ids = ", ".join(topic_sessions[topic])
        ax.text(0.05, i, session_ids, va="center", fontsize=8,
                color=BG_DARK, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Risk Assessment Visualization
# ---------------------------------------------------------------------------

def generate_risk_assessment(sessions: list[dict], save_path: Path) -> None:
    """Generate a risk assessment visualization across sessions."""
    risk_order = {"none": 0, "low": 1, "moderate": 2, "high": 3, "imminent": 4}
    risk_labels = {"none": "None", "low": "Low", "moderate": "Moderate",
                   "high": "High", "imminent": "Imminent"}

    labels = [s["label"] for s in sessions]
    risk_levels = [s["risk_level"] for s in sessions]
    risk_nums = [risk_order[r] for r in risk_levels]
    colors = [RISK_COLORS[r] for r in risk_levels]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    bars = ax.bar(range(len(labels)), risk_nums, color=colors,
                  edgecolor="none", width=0.65, alpha=0.85)

    # Add risk level text on each bar
    for i, (bar, level) in enumerate(zip(bars, risk_levels)):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.08,
                risk_labels[level],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=RISK_COLORS[level])

    # Add session type below bars
    for i, s in enumerate(sessions):
        short_type = s["topic"].split("(")[0].strip()
        if len(short_type) > 14:
            short_type = short_type[:13] + "."
        ax.text(i, -0.35, short_type, ha="center", va="top",
                fontsize=7.5, color=TEXT_LIGHT, rotation=30)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["None", "Low", "Moderate", "High", "Imminent"], fontsize=10)
    ax.set_ylabel("Risk Level", fontsize=12)
    ax.set_title("Risk Assessment Across 12 Therapy Sessions",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(axis="y", alpha=0.2)

    # Legend
    legend_handles = [
        mpatches.Patch(color=RISK_COLORS[k], label=v)
        for k, v in risk_labels.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.3, edgecolor=GRID_COLOR)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: Retrieval Scores Bar Chart
# ---------------------------------------------------------------------------

def generate_retrieval_scores(save_path: Path) -> None:
    """Generate a bar chart of retrieval similarity scores."""
    data = load_retrieval_data()
    if not data or "results" not in data:
        print("  WARNING: retrieval_example.json not found, using fallback data")
        results = []
    else:
        results = data["results"]

    chunk_ids = [r["chunk_id"].replace("session_", "S").replace("_chunk_", " #")
                 for r in results]
    scores = [r["similarity_score"] for r in results]
    session_types = []
    for r in results:
        stype = r.get("metadata", {}).get("session_type", "")
        short = stype.split("—")[-1].strip() if "—" in stype else stype
        session_types.append(short)

    # Colors gradient from high to low
    colors_list = [PRIMARY, ACCENT_1, ACCENT_2, "#d4a843", "#7f8c8d"]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(range(len(chunk_ids)), scores, color=colors_list[:len(scores)],
                   edgecolor="none", height=0.6, alpha=0.85)

    ax.set_yticks(range(len(chunk_ids)))
    ax.set_yticklabels(chunk_ids, fontsize=10)
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_title("Retrieval Results: \"PTSD symptoms including nightmares and hypervigilance\"",
                 fontsize=12, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.2)

    # Score labels
    for bar, score, stype in zip(bars, scores, session_types):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)
        ax.text(0.02, bar.get_y() + bar.get_height() / 2,
                stype, va="center", fontsize=8, color=BG_DARK, fontweight="bold")

    # Query metadata
    meta = data.get("metadata", {})
    if meta:
        meta_text = (f"Collection: {meta.get('collection_size', '?')} chunks  |  "
                     f"Model: {meta.get('embedding_model', '?')}  |  "
                     f"Metric: {meta.get('similarity_metric', '?')}  |  "
                     f"Latency: {meta.get('query_time_ms', '?')}ms")
        ax.text(0.5, -0.12, meta_text,
                transform=ax.transAxes, ha="center", fontsize=9,
                color=TEXT_LIGHT, style="italic")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 5: API Flow Diagram
# ---------------------------------------------------------------------------

def generate_api_flow(save_path: Path) -> None:
    """Generate the FastAPI endpoint flow diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)

    # Title
    ax.text(6.0, 6.6, "FastAPI Endpoint Flow",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TEXT_COLOR)

    # --- POST /summarize flow (main flow) ---
    flow_boxes = [
        ("Client\nRequest", 1.5, 5.2, ACCENT_1, 1.8, 0.9),
        ("POST\n/summarize", 4.0, 5.2, PRIMARY, 1.8, 0.9),
        ("Pydantic\nValidation", 6.5, 5.2, ACCENT_2, 1.8, 0.9),
        ("Extract\nMetadata", 9.0, 5.2, PRIMARY, 1.8, 0.9),
    ]

    flow_boxes_2 = [
        ("Retrieve Similar\nSessions", 2.5, 3.3, ACCENT_1, 2.0, 0.9),
        ("Risk\nAssessment", 5.5, 3.3, "#d4a843", 1.8, 0.9),
        ("LLM\nSummarization", 8.5, 3.3, ACCENT_2, 2.0, 0.9),
    ]

    output_box = ("SessionSummary\nJSON Response", 6.0, 1.5, PRIMARY, 2.4, 0.9)

    # Draw all boxes
    all_boxes = flow_boxes + flow_boxes_2 + [output_box]
    for label, x, y, color, w, h in all_boxes:
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color + "25",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=TEXT_COLOR)

    # Arrows for top row
    for i in range(len(flow_boxes) - 1):
        x1 = flow_boxes[i][1] + flow_boxes[i][4] / 2
        x2 = flow_boxes[i + 1][1] - flow_boxes[i + 1][4] / 2
        y = flow_boxes[i][2]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color=TEXT_LIGHT, lw=1.5))

    # Arrow from Extract Metadata down to row 2
    ax.annotate("", xy=(5.5, 3.3 + 0.45), xytext=(9.0, 5.2 - 0.45),
                 arrowprops=dict(arrowstyle="->", color=TEXT_LIGHT, lw=1.5,
                                  connectionstyle="arc3,rad=0.3"))

    # Arrows between row 2 boxes
    for i in range(len(flow_boxes_2) - 1):
        x1 = flow_boxes_2[i][1] + flow_boxes_2[i][4] / 2
        x2 = flow_boxes_2[i + 1][1] - flow_boxes_2[i + 1][4] / 2
        y = flow_boxes_2[i][2]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color=TEXT_LIGHT, lw=1.5))

    # Arrow from LLM to output
    ax.annotate("", xy=(output_box[1], output_box[2] + output_box[5] / 2),
                 xytext=(8.5, 3.3 - 0.45),
                 arrowprops=dict(arrowstyle="->", color=TEXT_LIGHT, lw=1.5,
                                  connectionstyle="arc3,rad=0.3"))

    # Side endpoints
    side_endpoints = [
        ("GET /health", 1.0, 1.5, ACCENT_1),
        ("GET /sessions", 10.5, 1.5, ACCENT_2),
    ]
    for label, x, y, color in side_endpoints:
        box = FancyBboxPatch(
            (x - 0.9, y - 0.35), 1.8, 0.7,
            boxstyle="round,pad=0.1",
            facecolor=color + "20",
            edgecolor=color,
            linewidth=1.5,
            linestyle="dashed",
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

    # ChromaDB callout
    chroma_x, chroma_y = 2.5, 1.5
    box = FancyBboxPatch(
        (chroma_x - 0.9, chroma_y - 0.35), 1.8, 0.7,
        boxstyle="round,pad=0.1",
        facecolor="#2a4a3a",
        edgecolor=PRIMARY,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(chroma_x, chroma_y, "ChromaDB",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=PRIMARY)

    ax.annotate("", xy=(chroma_x, chroma_y + 0.35),
                 xytext=(2.5, 3.3 - 0.45),
                 arrowprops=dict(arrowstyle="<->", color=PRIMARY, lw=1.2,
                                  linestyle="dashed"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all figures."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures for Mental Health RAG project...\n")

    # Extract data from transcripts
    sessions = extract_session_data()
    print(f"  Loaded {len(sessions)} session transcripts")

    # Generate all figures
    generate_rag_architecture(IMAGES_DIR / "rag_architecture.png")
    generate_session_topics(sessions, IMAGES_DIR / "session_topics.png")
    generate_risk_assessment(sessions, IMAGES_DIR / "risk_assessment.png")
    generate_retrieval_scores(IMAGES_DIR / "retrieval_scores.png")
    generate_api_flow(IMAGES_DIR / "api_flow.png")

    print(f"\nAll figures saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
