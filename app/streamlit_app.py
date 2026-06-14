import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lite.config import load_config, ensure_project_dirs
from rag_lite.pipeline import RagLitePipeline

load_dotenv()

st.set_page_config(
    page_title="RAG-LITE-LR",
    layout="wide",
)

config = load_config()
ensure_project_dirs(config)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("RAG-LITE-LR")
    st.caption(
        "Lightweight literature search: HyDE + dense (SBERT) + BM25 "
        "retrieval, fused with RRF, with Claude relevance notes."
    )

    domain_options = {
        "All domains": None,
        "Education": "education",
        "Environment & Conservation": "environment",
        "Research": "research",
        "Arts & Culture": "arts_culture",
    }
    domain_label = st.selectbox(
        "Research domain",
        list(domain_options.keys()),
        help="Restrict the search to documents tagged with this domain, or search the whole collection.",
    )
    domain = domain_options[domain_label]

    default_top_k = config["retrieval"].get("default_top_k", 5)
    top_k = st.slider("Number of results", min_value=1, max_value=15, value=default_top_k)
    use_hyde = st.checkbox("Use HyDE (hypothetical document)", value=config["retrieval"].get("use_hyde", True))
    use_justification = st.checkbox("Claude relevance notes", value=config["retrieval"].get("use_justification", True))

# -----------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------
st.header("Search your document collection")

query = st.text_area(
    "What are you looking for?",
    placeholder="e.g. What programs improve early literacy outcomes for low-income students?",
    height=100,
)

search_button = st.button("Search", type="primary")

if search_button:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        if st.session_state.pipeline is None:
            with st.spinner("Loading pipeline (first load may take a minute) ..."):
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
                f"Searched {response.trace.candidate_pool_size:,} documents in the "
                f"“{domain_label}” domain (of {response.trace.total_corpus_size:,} total) "
                f"in {response.trace.total_latency_seconds:.1f}s"
            )
        else:
            st.caption(f"Searched {response.trace.total_corpus_size:,} documents in {response.trace.total_latency_seconds:.1f}s")

        st.divider()
        st.subheader(f"Top {len(response.results)} results")

        if not response.results:
            st.warning("No results found. Try a broader or differently-worded query.")

        rank_by_id = {r.doc_id: r.rank for r in response.results}

        for r in response.results:
            with st.container():
                st.markdown(f"#### {r.rank}. {r.title}")
                col_a, col_b = st.columns(2)
                col_a.caption(f"**Year:** {r.year}  ·  **Source:** {r.source or '-'}  ·  **Domain:** {r.domain or '-'}")
                col_b.caption(
                    f"**Fusion score:** {r.rrf_score:.4f}",
                    help=(
                        "Reciprocal Rank Fusion score — an ordinal value used to "
                        "order these results, not a relevance probability. Use "
                        f"the result's position (#{r.rank}) for ranking."
                    ),
                )

                if r.relevance_justification:
                    st.info(f"**Why relevant:** {r.relevance_justification}")

                if r.contribution:
                    st.caption(f"**Summary:** {r.contribution}")

                scores = []
                if r.relevance_score is not None:
                    scores.append(f"Relevance: {r.relevance_score}/10")
                if r.specificity_score is not None:
                    scores.append(f"Specificity: {r.specificity_score}/10")
                if scores:
                    st.caption(" · ".join(scores))

                st.write(r.abstract_snippet + ("..." if len(r.abstract_snippet) >= 500 else ""))

                if r.possible_duplicate_of:
                    dup_ranks = sorted(
                        rank_by_id[did] for did in r.possible_duplicate_of if did in rank_by_id
                    )
                    if dup_ranks:
                        dup_labels = ", ".join(f"#{n}" for n in dup_ranks)
                        st.caption(
                            f":large_yellow_circle: **Possible near-duplicate** of result "
                            f"{dup_labels} — very similar title/abstract.",
                        )

                if r.url:
                    st.markdown(f"[View source →]({r.url})")

                st.divider()

        st.download_button(
            label="Download results as JSON",
            data=response.model_dump_json(indent=2),
            file_name="search_results.json",
            mime="application/json",
        )
