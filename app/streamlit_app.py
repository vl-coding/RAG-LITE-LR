import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lite.config import load_config, ensure_project_dirs
from rag_lite.pipeline import RagLitePipeline

load_dotenv()

# On Streamlit Community Cloud, secrets are configured via the app dashboard
# (or .streamlit/secrets.toml locally) and aren't exported to the environment
# automatically. Bridge them so the Anthropic SDK can pick up the key.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except st.errors.StreamlitSecretNotFoundError:
    pass

st.set_page_config(
    page_title="RAG-LITE-LR",
    page_icon="🔎",
    layout="wide",
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY is not set. Add it to your environment (.env locally) "
        "or to this app's Secrets in Streamlit Community Cloud before searching."
    )
    st.stop()

config = load_config()
ensure_project_dirs(config)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# -----------------------------------------------------------------------
# Domains: plain-language labels, friendly icons, and an accent color used
# to theme the page once a domain is chosen.
# -----------------------------------------------------------------------
DOMAINS = {
    "All topics": {
        "key": None, "icon": "🔎", "color": "#1F6FEB",
        "tagline": "Search everything in the collection.",
        "example_query": "What programs improve early literacy outcomes for low-income students?",
    },
    "Education": {
        "key": "education", "icon": "📚", "color": "#2563EB",
        "tagline": "Programs, curricula, and learning outcomes.",
        "example_query": "What programs improve early literacy outcomes for low-income students?",
    },
    "Environment & Conservation": {
        "key": "environment", "icon": "🌱", "color": "#15803D",
        "tagline": "Sustainability, climate, and conservation work.",
        "example_query": "What strategies help reduce carbon emissions in urban communities?",
    },
    "Research": {
        "key": "research", "icon": "🔬", "color": "#7C3AED",
        "tagline": "Studies, evaluations, and methodology.",
        "example_query": "What evaluation methods are used to measure long-term program impact?",
    },
    "Arts & Culture": {
        "key": "arts_culture", "icon": "🎨", "color": "#DB2777",
        "tagline": "Community arts, heritage, and cultural programs.",
        "example_query": "How do community arts programs increase access for underserved neighborhoods?",
    },
}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def apply_theme(color: str) -> None:
    """Recolor key Streamlit widgets to match the chosen domain."""
    soft = hex_to_rgba(color, 0.08)
    page_bg = hex_to_rgba(color, 0.12)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {page_bg};
        }}
        .stButton > button[kind="primary"] {{
            background-color: {color};
            border-color: {color};
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {color};
            opacity: 0.85;
            border-color: {color};
        }}
        div[data-baseweb="slider"] div[role="slider"] {{
            background-color: {color} !important;
            border-color: {color} !important;
        }}
        div[data-baseweb="slider"] div[style*="height: 0.25rem"] {{
            background-image: none !important;
            background-color: {color} !important;
        }}
        div[data-testid="stSliderThumbValue"] {{
            color: {color} !important;
        }}
        label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {{
            background-color: {color} !important;
            border-color: {color} !important;
        }}
        .app-banner {{
            background-color: {soft};
            border-left: 6px solid {color};
            padding: 0.9rem 1.1rem;
            border-radius: 0.5rem;
            margin-bottom: 1.2rem;
        }}
        div[class*="st-key-result_"] {{
            border-left: 5px solid {color};
            background-color: {soft};
            padding: 1rem 1.2rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 RAG-LITE-LR")
    st.caption("Find relevant reports and research from your organization's document library.")

    st.markdown("#### 1. What are you looking for?")
    domain_label = st.selectbox(
        "Topic area",
        list(DOMAINS.keys()),
        help="Narrow your search to a specific topic area, or search everything.",
    )
    domain_info = DOMAINS[domain_label]
    domain = domain_info["key"]
    st.caption(domain_info["tagline"])

    st.markdown("#### 2. Search settings")
    default_top_k = config["retrieval"].get("default_top_k", 5)
    top_k = st.slider(
        "How many results to show",
        min_value=1, max_value=15, value=default_top_k,
    )
    use_hyde = st.checkbox(
        "Smarter search (recommended)",
        value=config["retrieval"].get("use_hyde", True),
        help="Helps the search understand what you're looking for, even if your wording doesn't exactly match the documents.",
    )
    use_justification = st.checkbox(
        "Explain why each result matches",
        value=config["retrieval"].get("use_justification", True),
        help="Adds a short, plain-language note explaining why each result was included.",
    )

    st.divider()
    st.caption("Tip: describe what you need in everyday language — full sentences and questions work well.")

apply_theme(domain_info["color"])

# -----------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-banner">
        <h2 style="margin:0;">{domain_info['icon']} Searching: {domain_label}</h2>
        <p style="margin:0.25rem 0 0 0;">{domain_info['tagline']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

query = st.text_area(
    "What would you like to find?",
    placeholder=f"e.g. {domain_info['example_query']}",
    height=100,
)

search_button = st.button("Search", type="primary")

if search_button:
    if not query.strip():
        st.warning("Please describe what you're looking for first.")
    else:
        if st.session_state.pipeline is None:
            with st.spinner("Getting things ready (this may take a minute the first time) ..."):
                st.session_state.pipeline = RagLitePipeline(config)

        progress_bar = st.progress(0, text="Starting search ...")

        def _update_progress(step: str, fraction: float) -> None:
            progress_bar.progress(fraction, text=step)

        response = st.session_state.pipeline.run(
            query=query,
            top_k=top_k,
            use_hyde=use_hyde,
            use_justification=use_justification,
            domain=domain,
            progress_callback=_update_progress,
        )

        progress_bar.empty()

        if response.trace.candidate_pool_size is not None:
            st.caption(
                f"Looked through {response.trace.candidate_pool_size:,} documents in "
                f"{domain_info['icon']} {domain_label} (out of {response.trace.total_corpus_size:,} total) "
                f"in {response.trace.total_latency_seconds:.1f} seconds."
            )
        else:
            st.caption(
                f"Looked through {response.trace.total_corpus_size:,} documents "
                f"in {response.trace.total_latency_seconds:.1f} seconds."
            )

        st.divider()

        if not response.results:
            st.warning("No results found. Try describing your topic differently, or search a broader topic area.")
        else:
            st.subheader(f"Top {len(response.results)} results")

        rank_by_id = {r.doc_id: r.rank for r in response.results}

        for r in response.results:
            with st.container(key=f"result_{r.rank}"):
                st.markdown(f"#### {r.rank}. {r.title}")
                st.caption(f"{r.year}  ·  {r.source or 'Unknown source'}  ·  {r.domain or 'General'}")

                if r.relevance_justification:
                    st.info(f"**Why this matches:** {r.relevance_justification}")

                if r.contribution:
                    st.caption(f"**Summary:** {r.contribution}")

                st.write(r.abstract_snippet + ("..." if len(r.abstract_snippet) >= 500 else ""))

                if r.possible_duplicate_of:
                    dup_ranks = sorted(
                        rank_by_id[did] for did in r.possible_duplicate_of if did in rank_by_id
                    )
                    if dup_ranks:
                        dup_labels = ", ".join(f"#{n}" for n in dup_ranks)
                        st.caption(
                            f":large_yellow_circle: This looks very similar to result {dup_labels}.",
                        )

                if r.url:
                    st.markdown(f"[View full document →]({r.url})")

                if r.relevance_score is not None or r.specificity_score is not None:
                    with st.expander("More details"):
                        if r.relevance_score is not None:
                            st.caption(f"How well it matches your topic: {r.relevance_score}/10")
                        if r.specificity_score is not None:
                            st.caption(f"How specific it is to your question: {r.specificity_score}/10")
                        st.caption(
                            f"Search ranking score: {r.rrf_score:.4f} "
                            "(used to order results, not a percentage)"
                        )

        if response.results:
            st.download_button(
                label="Download these results",
                data=response.model_dump_json(indent=2),
                file_name="search_results.json",
                mime="application/json",
            )
