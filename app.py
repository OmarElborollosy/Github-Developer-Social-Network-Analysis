# dashboard/app.py
# GitHub Developer SNA — Streamlit Dashboard
# 4 tabs: Network Overview | Centrality | Communities | Link Prediction & Diffusion
# Run from the dashboard/ directory:  streamlit run app.py
#
# Fixed constants (spec rules):
#   random.seed(42)  np.random.seed(42)  at top of every computation
#   Betweenness: always k=500, seed=42
#   All figures: dpi=150, title, labeled axes, legend

import random
import numpy as np

random.seed(42)
np.random.seed(42)

import os
import sys
import tempfile

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ── Path resolution (works whether launched from dashboard/ or project root) ──
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
DATA_DIR  = os.path.join(PROJ_ROOT, 'data')
FIG_DIR   = os.path.join(PROJ_ROOT, 'outputs', 'figures')
RES_DIR   = os.path.join(PROJ_ROOT, 'outputs', 'results')

EDGES_PATH   = os.path.join(DATA_DIR, 'musae_git_edges.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'musae_git_target.csv')

# Make dashboard/utils importable regardless of cwd
sys.path.insert(0, PROJ_ROOT)
from dashboard import utils as U

# ── Image helper (reads as bytes so Streamlit serves on any OS) ─────────────
def _show_img(path, caption=''):
    """Display an image from an absolute path. No-op if file missing."""
    if os.path.exists(path):
        with open(path, 'rb') as _f:
            st.image(_f.read(), caption=caption)
    else:
        st.info(f'Figure not found: {os.path.basename(path)}')

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='GitHub SNA Dashboard',
    page_icon='🌐',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  h1  { color: #1565C0; letter-spacing: -0.5px; }
  h2  { color: #1976D2; }
  .stMetric [data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached resource: graph (non-serialisable → cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading GitHub graph — first run takes ~30 s …')
def get_graph():
    """Load edges + targets, build graph + LCC once per process lifetime."""
    G     = U.load_graph(EDGES_PATH, TARGETS_PATH)
    G_lcc = U.get_lcc(G)
    labels = nx.get_node_attributes(G_lcc, 'developer_type')
    return G, G_lcc, labels



# ─────────────────────────────────────────────────────────────────────────────
# Cached data: CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_network_stats():
    return pd.read_csv(os.path.join(RES_DIR, 'network_stats.csv'))

@st.cache_data
def get_centrality_df():
    return pd.read_csv(os.path.join(RES_DIR, 'centrality_scores.csv'))

@st.cache_data
def get_rank_corr():
    return pd.read_csv(os.path.join(RES_DIR, 'rank_correlation.csv'), index_col=0)

@st.cache_data
def get_community_composition():
    return pd.read_csv(os.path.join(RES_DIR, 'community_composition.csv'))

@st.cache_data
def get_link_prediction_results():
    return pd.read_csv(os.path.join(RES_DIR, 'link_prediction_results.csv'))

@st.cache_data
def get_diffusion_results():
    p = os.path.join(RES_DIR, 'diffusion_results.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

G, G_lcc, labels = get_graph()

# Preload all heavy CSVs at startup
get_network_stats()
get_centrality_df()
get_rank_corr()
get_community_composition()
get_link_prediction_results()
get_diffusion_results()


# ─────────────────────────────────────────────────────────────────────────────
# Cached: Tab 1 Pyvis network HTML (top 400 nodes, coloured by developer type)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_degree_sequence(_G):
    return sorted([d for _, d in _G.degree()], reverse=True)

@st.cache_resource(show_spinner='Building interactive network …')
def _tab1_pyvis_html():
    """
    Top 400 nodes by degree, coloured by developer type.
    ML = orange (#FF5722), Web = blue (#2196F3).
    Node hover shows ID and type.
    Returns raw HTML string.
    """
    from pyvis.network import Network

    top400 = [n for n, _ in sorted(G_lcc.degree(), key=lambda x: x[1], reverse=True)[:400]]
    G_sub  = G_lcc.subgraph(top400)

    net = Network(height='550px', width='100%',
                  bgcolor='#111827', font_color='white',
                  notebook=False)
    pos = nx.spring_layout(G_sub, seed=42, k=0.5)

    for node in G_sub.nodes():
        dev_type = labels.get(node, 0)
        color    = '#FF5722' if dev_type == 1 else '#2196F3'
        type_str = 'ML' if dev_type == 1 else 'Web'
        deg      = G_lcc.degree(node)
        net.add_node(
            node,
            label=str(node),
            color=color,
            size=max(6, min(25, deg // 5)),
            title=f'Node {node} | Type: {type_str} | Degree: {deg}',
            x=pos[node][0] * 1000,
            y=pos[node][1] * 1000,
            physics=False
        )

    for u, v in G_sub.edges():
        net.add_edge(u, v, color='rgba(160,160,160,0.25)', width=0.6)

    net.toggle_physics(False)
    net.set_options("""
    {
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)

    # Write to a temp file and read back as string
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False,
                                     mode='w', encoding='utf-8')
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, 'r', encoding='utf-8') as f:
        html = f.read()
    os.unlink(tmp.name)
    return html

# ─────────────────────────────────────────────────────────────────────────────
# Helper: developer-type colour column
# ─────────────────────────────────────────────────────────────────────────────
def _add_type_label(df, col='developer_type'):
    df = df.copy()
    df['Developer Type'] = df[col].map({0: 'Web', 1: 'ML'})
    return df

# ─────────────────────────────────────────────────────────────────────────────
# App header
# ─────────────────────────────────────────────────────────────────────────────
st.title('🌐 GitHub Developer Social Network Analysis Dashboard')
st.caption(
    'MUSAE GitHub Social Network (SNAP) · **37,700** developer nodes · '
    '**289,003** mutual-follow edges · Node label: 0 = Web developer, 1 = ML developer'
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📊 Network Overview',
    '🎯 Centrality Analysis',
    '🏘️ Community Detection',
    '🔗 Link Prediction & Diffusion',
    '⏱️ Temporal Analysis'
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Network Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Network Overview')

    # ── 6 metric cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric('Nodes',              '37,700')
    c2.metric('Edges',              '289,003')
    c3.metric('Density',            '0.000407')
    c4.metric('Avg Clustering',     '0.1675')
    c5.metric('Components',         '1')
    c6.metric('LCC Size',           '37,700')

    st.divider()

    # ── Developer-type breakdown ───────────────────────────────────────────────
    n_ml  = sum(1 for v in labels.values() if v == 1)
    n_web = sum(1 for v in labels.values() if v == 0)
    col_pie, col_stat = st.columns([1, 2])

    with col_pie:
        @st.cache_data
        def get_pie_chart(nw, nm):
            p = go.Figure(go.Pie(
                labels=['Web Developer', 'ML Developer'],
                values=[nw, nm],
                marker_colors=['#2196F3', '#FF5722'],
                hole=0.42,
                textinfo='label+percent',
            ))
            p.update_layout(title='Developer Type Split', height=280,
                              margin=dict(t=40, b=10, l=10, r=10),
                              showlegend=False)
            return p
        st.plotly_chart(get_pie_chart(n_web, n_ml), use_container_width=True)

    with col_stat:
        st.markdown('**Summary Statistics**')
        st.dataframe(get_network_stats().head(100), use_container_width=True, height=280)
        st.caption('Showing top 100 rows. Full data available in outputs/results/')

    st.divider()

    # ── Interactive Pyvis network: top 400 nodes, coloured by developer type ──
    st.markdown(
        '**Interactive Network — Top 400 developers by degree** '
        '(🟠 ML developer · 🔵 Web developer · hover for details)'
    )
    with st.spinner('Rendering interactive network …'):
        html = _tab1_pyvis_html()
    components.html(html, height=570, scrolling=False)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Centrality Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Centrality Analysis')

    cent_df = get_centrality_df()

    # ── Controls ───────────────────────────────────────────────────────────────
    col_dd, col_sl = st.columns([2, 1])
    with col_dd:
        measure = st.selectbox(
            'Select centrality measure',
            options=['degree', 'betweenness', 'closeness', 'eigenvector'],
            format_func=str.capitalize,
            key='cent_measure',
        )
    with col_sl:
        top_n = st.slider('Top N nodes', min_value=5, max_value=50, value=10,
                          key='cent_topn')

    # ── Results table ──────────────────────────────────────────────────────────
    tbl = (
        _add_type_label(cent_df)
        [['node', measure, 'Developer Type']]
        .rename(columns={'node': 'Node', measure: 'Score'})
        .nlargest(top_n, 'Score')
        .reset_index(drop=True)
    )
    tbl.index += 1
    st.dataframe(tbl.head(100), use_container_width=True)
    st.caption('Showing top 100 rows. Full data available in outputs/results/')

    # ── Bar chart coloured by developer type ───────────────────────────────────
    @st.cache_data
    def get_centrality_bar(table, meas, n):
        b = px.bar(
            table, x='Node', y='Score', color='Developer Type',
            color_discrete_map={'Web': '#2196F3', 'ML': '#FF5722'},
            title=f'Top {n} Nodes — {meas.capitalize()} Centrality',
            labels={'Score': meas.capitalize(), 'Node': 'Node ID'},
        )
        b.update_layout(xaxis_type='category', height=380)
        return b
    st.plotly_chart(get_centrality_bar(tbl, measure, top_n), use_container_width=True)

    st.divider()

    # ── Static figures + Spearman heatmap ─────────────────────────────────────
    col_h, col_t = st.columns(2)
    with col_h:
        _show_img(os.path.join(FIG_DIR, 'centrality_heatmap.png'),
                  caption='Centrality Correlation Heatmap')
    with col_t:
        _show_img(os.path.join(FIG_DIR, 'centrality_by_type.png'),
                  caption='Centrality Distributions by Developer Type')

    st.markdown('**Spearman Rank Correlation (interactive)**')
    rk = get_rank_corr()
    @st.cache_data
    def get_corr_fig(rank_data):
        cf = px.imshow(
            rank_data, color_continuous_scale='RdBu', zmin=-1, zmax=1,
            text_auto='.2f',
            title='Centrality Rank Correlation (Spearman ρ)',
        )
        cf.update_layout(height=400)
        return cf
    st.plotly_chart(get_corr_fig(rk), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Community Detection
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader('Community Detection — Louvain Algorithm')

    if st.button('▶ Run Louvain Community Detection', key='btn_louvain'):
        with st.spinner('Running Louvain on 37,700-node graph (~60 s) …'):
            partition, mod = U.run_louvain(G_lcc, random_state=42)
            st.session_state['partition']  = partition
            st.session_state['modularity'] = mod

    if 'partition' in st.session_state:
        p   = st.session_state['partition']
        mod = st.session_state['modularity']

        from collections import Counter
        sizes = Counter(p.values())

        m1, m2, m3 = st.columns(3)
        m1.metric('Communities Found', len(sizes))
        m2.metric('Modularity Score',  f'{mod:.4f}')
        m3.metric('Largest Community', f'{max(sizes.values()):,} nodes')

        # Community size bar chart — top 20
        size_df = (
            pd.DataFrame(sorted(sizes.items(), key=lambda x: -x[1]),
                         columns=['Community', 'Size'])
            .head(20)
        )
        @st.cache_data
        def get_comm_bar(sdf):
            cb = px.bar(
                sdf, x='Community', y='Size',
                title='Top 20 Community Sizes',
                color='Size', color_continuous_scale='Blues',
                labels={'Community': 'Community ID', 'Size': 'Number of Nodes'},
            )
            cb.update_layout(xaxis_type='category', height=380,
                                   coloraxis_showscale=False)
            return cb
        st.plotly_chart(get_comm_bar(size_df), use_container_width=True)
    else:
        st.info('Click **▶ Run Louvain Community Detection** above to compute communities.')

    st.divider()

    # ── Pre-computed static figures ────────────────────────────────────────────
    col_cs, col_cn = st.columns(2)
    with col_cs:
        _show_img(os.path.join(FIG_DIR, 'community_sizes.png'),
                  caption='Pre-computed: Community Size Distribution')
    with col_cn:
        _show_img(os.path.join(FIG_DIR, 'community_network.png'),
                  caption='Pre-computed: Network Coloured by Community')

    # ── Community composition table ────────────────────────────────────────────
    st.markdown('**Community Composition by Developer Type**')
    st.dataframe(get_community_composition().head(100), use_container_width=True)
    st.caption('Showing top 100 rows. Full data available in outputs/results/')

    # ── Pre-generated interactive Pyvis (coloured by community) ───────────────
    html_path = os.path.join(FIG_DIR, 'interactive_community.html')
    if os.path.exists(html_path):
        st.markdown('**Interactive Network — Top 400 nodes (coloured by community)**')
        with open(html_path, 'r', encoding='utf-8') as f:
            comm_html = f.read()
        components.html(comm_html, height=580, scrolling=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Link Prediction + PageRank + Diffusion
# ═════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── ① Pre-computed validated results ──────────────────────────────────────
    st.subheader('Link Prediction — Validated Results')
    st.caption('Computed in notebook Part D with strict no-leakage protocol (G_train only).')
    st.dataframe(get_link_prediction_results().head(100), use_container_width=True)
    st.caption('Showing top 100 rows. Full data available in outputs/results/')

    col_roc, col_auc = st.columns(2)
    with col_roc:
        _show_img(os.path.join(FIG_DIR, 'link_prediction_roc.png'),
                  caption='ROC Curves — all 4 methods')
    with col_auc:
        _show_img(os.path.join(FIG_DIR, 'link_pred_type_auc.png'),
                  caption='AUC by Developer Pair Type')

    st.divider()

    # ── ② Live link prediction button ─────────────────────────────────────────
    st.subheader('Run Live Link Prediction')
    st.info(
        '10% random edge hold-out · negatives sampled from true non-edges of G_train · '
        'no data leakage — all 4 scores computed on G_train only.'
    )

    if st.button('▶ Run All 4 Link Prediction Methods', key='btn_lp'):
        with st.spinner('Running link prediction (may take 2–5 min) …'):
            random.seed(42); np.random.seed(42)
            lp = U.run_link_prediction(G_lcc, test_frac=0.1, seed=42)
            st.session_state['lp_result'] = lp

    if 'lp_result' in st.session_state:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

        lp = st.session_state['lp_result']
        rows = []
        for name, scores in lp['scores'].items():
            rows.append({
                'Method':        name,
                'AUC-ROC':       round(roc_auc_score(lp['y_true'], scores), 4),
                'Avg Precision': round(average_precision_score(lp['y_true'], scores), 4),
            })
        live_df = pd.DataFrame(rows).sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
        st.dataframe(live_df.head(100), use_container_width=True)
        st.caption('Showing top 100 rows. Full data available in outputs/results/')

        # Interactive ROC curves (Plotly)
        PALETTE = {
            'Common Neighbors':    '#E74C3C',
            'Jaccard Coefficient': '#3498DB',
            'Adamic-Adar':         '#2ECC71',
            'Resource Allocation': '#F39C12',
        }
        @st.cache_data
        def get_roc_fig(lp_data):
            rf = go.Figure()
            for name, scores in lp_data['scores'].items():
                fpr, tpr, _ = roc_curve(lp_data['y_true'], scores)
                auc = roc_auc_score(lp_data['y_true'], scores)
                rf.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{name} (AUC={auc:.3f})',
                    line=dict(color=PALETTE.get(name, '#888'), width=2),
                ))
            rf.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash='dash', color='gray', width=1))
            rf.update_layout(
                title='ROC Curves — Link Prediction Methods',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(x=0.55, y=0.05),
                height=420,
            )
            return rf
        st.plotly_chart(get_roc_fig(lp), use_container_width=True)

    st.divider()

    # ── ③ PageRank top-N bar chart ─────────────────────────────────────────────
    st.subheader('PageRank — Top Influencers')

    pr_n = st.slider('Top N by PageRank', min_value=5, max_value=50, value=10, key='sl_pr')
    cent_tab4 = _add_type_label(get_centrality_df())
    pr_disp   = (
        cent_tab4[['node', 'pagerank', 'Developer Type']]
        .rename(columns={'node': 'Node', 'pagerank': 'PageRank'})
        .nlargest(pr_n, 'PageRank')
        .reset_index(drop=True)
    )
    @st.cache_data
    def get_pr_fig(pd_disp, p_n):
        pf = px.bar(
            pd_disp, x='Node', y='PageRank', color='Developer Type',
            color_discrete_map={'Web': '#2196F3', 'ML': '#FF5722'},
            title=f'Top {p_n} Most Influential Developers (PageRank)',
            labels={'Node': 'Node ID', 'PageRank': 'PageRank Score'},
        )
        pf.update_layout(xaxis_type='category', height=380)
        return pf
    st.plotly_chart(get_pr_fig(pr_disp, pr_n), use_container_width=True)

    # PPR comparison
    col_ml, col_wb = st.columns(2)
    with col_ml:
        st.markdown('**Personalized PageRank — ML-seeded (top 10)**')
        st.dataframe(
            _add_type_label(get_centrality_df()).head(100)
            [['node', 'ppr_ml', 'Developer Type']]
            .rename(columns={'node': 'Node', 'ppr_ml': 'PPR-ML'})
            .nlargest(10, 'PPR-ML')
            .reset_index(drop=True),
            use_container_width=True,
        )
        st.caption('Showing top 100 rows. Full data available in outputs/results/')
    with col_wb:
        st.markdown('**Personalized PageRank — Web-seeded (top 10)**')
        st.dataframe(
            _add_type_label(get_centrality_df()).head(100)
            [['node', 'ppr_web', 'Developer Type']]
            .rename(columns={'node': 'Node', 'ppr_web': 'PPR-Web'})
            .nlargest(10, 'PPR-Web')
            .reset_index(drop=True),
            use_container_width=True,
        )
        st.caption('Showing top 100 rows. Full data available in outputs/results/')

    st.divider()

    # ── ④ Information Diffusion Simulation (Part D2) ──────────────────────────
    st.subheader('📡 Information Diffusion Simulation')
    st.markdown(
        'Simulate information spread through the GitHub follow network using the '
        '**Independent Cascade (IC)** or **Linear Threshold (LT)** model '
        'seeded from a chosen centrality-based strategy.'
    )

    STRAT_COL = {
        'PageRank':    'pagerank',
        'PPR-ML':      'ppr_ml',
        'PPR-Web':     'ppr_web',
        'Degree':      'degree',
        'Betweenness': 'betweenness',
    }

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        k_sim = st.slider('Seed set size (k)', min_value=1, max_value=50,
                          value=10, key='sl_k_sim')
    with dc2:
        strat_sim = st.selectbox('Seed strategy',
                                 list(STRAT_COL.keys()), key='sel_strat')
    with dc3:
        model_sim = st.selectbox('Cascade model',
                                 ['IC (Independent Cascade)', 'LT (Linear Threshold)'],
                                 key='sel_model')

    if st.button('▶ Run Simulation', key='btn_diff'):
        with st.spinner(f'Running {model_sim} · strategy={strat_sim} · k={k_sim} …'):
            random.seed(42); np.random.seed(42)
            cent_sim  = get_centrality_df()
            col_name  = STRAT_COL[strat_sim]
            seed_nodes = (cent_sim
                          .sort_values(col_name, ascending=False)['node']
                          .tolist()[:k_sim])

            if model_sim.startswith('IC'):
                activated = U.run_ic_cascade(G_lcc, seed_nodes,
                                             prob=0.1, seed=42, max_rounds=20)
            else:
                activated = U.run_lt_cascade(G_lcc, seed_nodes, seed=42)

            st.session_state['diff_result'] = {
                'activated': activated,
                'strategy':  strat_sim,
                'model':     model_sim,
                'k':         k_sim,
                'total':     G_lcc.number_of_nodes(),
            }

    # Persist & display results
    if 'diff_result' in st.session_state:
        dr         = st.session_state['diff_result']
        act_size   = len(dr['activated'])
        reach_pct  = act_size / dr['total'] * 100

        rm1, rm2 = st.columns(2)
        rm1.metric('Cascade Size',   f'{act_size:,} nodes')
        rm2.metric('Network Reach',  f'{reach_pct:.2f} %')

        @st.cache_data
        def get_reach_fig(dr_data, a_size):
            rf = go.Figure(go.Bar(
                x=[a_size, dr_data['total'] - a_size],
                y=['Activated', 'Not Activated'],
                orientation='h',
                marker_color=['#E74C3C', '#95A5A6'],
                text=[f'{a_size:,}', f'{dr_data["total"]-a_size:,}'],
                textposition='auto',
            ))
            rf.update_layout(
                title=(f"Cascade Reach — {dr_data['strategy']} seeds · "
                       f"k={dr_data['k']} · {dr_data['model']}"),
                xaxis_title='Number of Developer Nodes',
                height=280,
                margin=dict(t=50, b=40),
            )
            return rf
        st.plotly_chart(get_reach_fig(dr, act_size), use_container_width=True)

    # Pre-computed static figures (shown once notebook D2 has been run)
    st.divider()
    st.markdown('**Pre-computed Diffusion Analysis** *(generated in notebook Part D2)*')

    _FIG_PATHS = {
        'diffusion_cascade_size.png':  'Cascade size vs seed set size — IC and LT models',
        'diffusion_seed_overlap.png':  'Seed set overlap (Jaccard similarity) at k=10',
        'diffusion_divergence.png':    'Cascade distribution: PageRank-unique vs Degree-unique seeds',
    }
    any_found = False
    for fname, caption in _FIG_PATHS.items():
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            _show_img(fpath, caption=caption)
            any_found = True
    if not any_found:
        st.info(
            'Diffusion figures not found. '
            'Run Part D2 cells in notebooks/analysis.ipynb to generate them, '
            'then refresh this page.'
        )

    # Pre-computed diffusion CSV
    diff_csv = get_diffusion_results()
    if diff_csv is not None:
        st.markdown('**Diffusion Simulation Results Table**')
        st.dataframe(diff_csv.head(100), use_container_width=True)
        st.caption('Showing top 100 rows. Full data available in outputs/results/')

# ── TAB 5: Temporal Analysis ──────────────────────────────────────────────
with tab5:
    st.header('⏱️ Temporal Analysis — GH Archive (June 1–7, 2019)')
    st.info(
        "This tab presents temporal analysis using an independent GH Archive "
        "co-starring interaction graph built from the same period as the MUSAE "
        "dataset (June 2019). Node identities differ from MUSAE — this serves "
        "as cross-dataset validation of the structural findings."
    )

    # ── Check if Part E outputs exist ──────────────────────────────────────
    temporal_edges_df   = U.load_temporal_edges()
    evolution_df        = U.load_temporal_network_stats()
    temporal_lp_df, comparison_df = U.load_temporal_link_prediction()

    if temporal_edges_df is None:
        st.warning(
            "Part E outputs not found. Run the Part E cells in "
            "notebooks/analysis.ipynb first to generate the temporal data."
        )
        st.stop()

    # ── Section 1: Temporal Graph Summary ─────────────────────────────────
    st.subheader('1. Temporal Graph Summary')

    unique_nodes = pd.concat([
        temporal_edges_df['source'],
        temporal_edges_df['target']
    ]).nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Temporal Nodes',  f'{unique_nodes:,}')
    col2.metric('Temporal Edges',  f'{len(temporal_edges_df):,}')
    col3.metric('Date Range',      'Jun 1–7, 2019')
    col4.metric('Edge Type',       'Co-starring / Co-forking')

    # Domain breakdown
    if 'domain' in temporal_edges_df.columns:
        domain_counts = temporal_edges_df['domain'].value_counts().reset_index()
        domain_counts.columns = ['Domain', 'Edge Count']
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Edges by Domain**")
            st.dataframe(domain_counts, use_container_width=True)
        with col6:
            fig_domain = px.pie(
                domain_counts, names='Domain', values='Edge Count',
                color='Domain',
                color_discrete_map={'ML': '#FF5722', 'Web': '#2196F3'},
                title='Temporal Edge Distribution by Domain'
            )
            st.plotly_chart(fig_domain, use_container_width=True)

    st.markdown("---")

    # ── Section 2: Daily Network Evolution ────────────────────────────────
    st.subheader('2. Daily Network Evolution')

    if evolution_df is not None:
        # Summary metrics from final day
        final = evolution_df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Final Day Nodes',       f'{int(final["nodes"]):,}')
        col2.metric('Final Day Edges',       f'{int(final["edges"]):,}')
        col3.metric('Final Modularity',      f'{final["modularity"]:.4f}')
        col4.metric('Final Communities',     f'{int(final["num_communities"])}')

        # Interactive Plotly evolution chart — 4 subplots
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig_evo = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cumulative Nodes', 'Avg Clustering Coefficient',
                'Modularity Q',     'Number of Communities'
            ]
        )
        fig_evo.add_trace(
            go.Scatter(x=evolution_df['date'], y=evolution_df['nodes'],
                       mode='lines+markers', name='Nodes',
                       line=dict(color='#2196F3', width=2)),
            row=1, col=1
        )
        fig_evo.add_trace(
            go.Scatter(x=evolution_df['date'], y=evolution_df['avg_clustering'],
                       mode='lines+markers', name='Clustering',
                       line=dict(color='#FF5722', width=2)),
            row=1, col=2
        )
        fig_evo.add_trace(
            go.Scatter(x=evolution_df['date'], y=evolution_df['modularity'],
                       mode='lines+markers', name='Modularity',
                       line=dict(color='#2ECC71', width=2)),
            row=2, col=1
        )
        fig_evo.add_trace(
            go.Scatter(x=evolution_df['date'], y=evolution_df['num_communities'],
                       mode='lines+markers', name='Communities',
                       line=dict(color='#9B59B6', width=2)),
            row=2, col=2
        )
        fig_evo.update_layout(
            title_text='GitHub Temporal Interaction Network — Daily Evolution',
            height=500, showlegend=False
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        # Raw data table
        with st.expander('View raw daily statistics table'):
            st.dataframe(evolution_df, use_container_width=True)
    else:
        st.image('outputs/figures/temporal_evolution.png',
                 caption='Daily network evolution (pre-computed)',
                 use_column_width=True)

    st.markdown("---")

    # ── Section 3: Burst Activity Detection ───────────────────────────────
    st.subheader('3. Burst Activity Detection')

    daily_counts_path = 'outputs/results/daily_edge_counts.csv'
    if os.path.exists(daily_counts_path):
        daily_df = pd.read_csv(daily_counts_path)

        mean_e  = daily_df['new_edges'].mean()
        std_e   = daily_df['new_edges'].std()
        thresh  = mean_e + 1.5 * std_e

        burst_days = daily_df[daily_df['new_edges'] > thresh]

        col1, col2, col3 = st.columns(3)
        col1.metric('Mean Daily Edges',    f'{mean_e:.0f}')
        col2.metric('Burst Threshold',     f'{thresh:.0f}')
        col3.metric('Burst Days Detected', f'{len(burst_days)}')

        # Interactive Plotly burst bar chart
        colors = ['#E74C3C' if b else '#3498DB'
                  for b in daily_df['new_edges'] > thresh]
        fig_burst = go.Figure()
        fig_burst.add_trace(go.Bar(
            x=daily_df['date'], y=daily_df['new_edges'],
            marker_color=colors, name='Daily Edges'
        ))
        fig_burst.add_hline(
            y=thresh, line_dash='dash', line_color='black',
            annotation_text=f'Burst threshold = {thresh:.0f}'
        )
        fig_burst.add_hline(
            y=mean_e, line_dash='dot', line_color='gray',
            annotation_text=f'Mean = {mean_e:.0f}'
        )
        fig_burst.update_layout(
            title='Daily New Edge Formation — Burst Detection',
            xaxis_title='Date', yaxis_title='New Edges',
            height=400
        )
        st.plotly_chart(fig_burst, use_container_width=True)

        if len(burst_days) > 0:
            st.success(f"Burst days detected: {burst_days['date'].tolist()}")
        else:
            st.info("No burst days detected in this 7-day window — "
                    "edge formation was uniform across the sample period.")
    else:
        st.image('outputs/figures/burst_activity.png',
                 caption='Burst activity detection (pre-computed)',
                 use_column_width=True)

    st.markdown("---")

    # ── Section 4: Temporal Link Prediction ───────────────────────────────
    st.subheader('4. Temporal Link Prediction')
    st.caption(
        "Train: June 1–5 | Test: June 6–7 | "
        "All features computed on training graph only (no data leakage)"
    )

    if temporal_lp_df is not None:
        # Results table
        st.markdown("**Temporal Link Prediction Results:**")
        st.dataframe(
            temporal_lp_df.style.highlight_max(
                subset=['Temporal_AUC'], color='#D5F5E3'
            ),
            use_container_width=True
        )

        # Comparison table with MUSAE
        if comparison_df is not None:
            st.markdown("**Comparison: Temporal vs MUSAE Random Hold-out:**")

            # Color delta column — green if positive, red if negative
            def color_delta(val):
                color = '#D5F5E3' if val >= 0 else '#FADBD8'
                return f'background-color: {color}'

            # Note: Pandas Styler applymap was renamed to map in newer Pandas, 
            # but I will use the user's provided code `.applymap` and gracefully handle it
            # actually the user requested `style.applymap`
            styled = comparison_df[
                ['Method', 'MUSAE_AUC', 'Temporal_AUC', 'Delta']
            ]
            if hasattr(styled.style, 'map'):
                styled = styled.style.map(color_delta, subset=['Delta'])
            else:
                styled = styled.style.applymap(color_delta, subset=['Delta'])

            st.dataframe(styled, use_container_width=True)

            # Interpretation
            avg_delta = comparison_df['Delta'].mean()
            if avg_delta < -0.03:
                st.warning(
                    f"Average AUC drop of {abs(avg_delta):.4f} under temporal "
                    "splitting confirms the network evolves over time — past "
                    "topology is a weaker predictor of future connections than "
                    "random hold-out implies."
                )
            elif avg_delta > 0.03:
                st.success(
                    f"Average AUC gain of {avg_delta:.4f} under temporal "
                    "splitting suggests strong structural regularity — the "
                    "network's collaborative patterns persist and strengthen "
                    "over time."
                )
            else:
                st.info(
                    f"Average AUC difference of {avg_delta:.4f} — comparable "
                    "performance across random and temporal splits confirms "
                    "structural regularity persists over short time horizons."
                )

        # Interactive ROC-style bar chart
        if comparison_df is not None:
            fig_comp = go.Figure()
            x = comparison_df['Method']
            fig_comp.add_trace(go.Bar(
                name='MUSAE — Random Hold-out',
                x=x, y=comparison_df['MUSAE_AUC'],
                marker_color='#2196F3', opacity=0.85
            ))
            fig_comp.add_trace(go.Bar(
                name='GH Archive — Temporal Split',
                x=x, y=comparison_df['Temporal_AUC'],
                marker_color='#FF5722', opacity=0.85
            ))
            fig_comp.add_hline(
                y=0.5, line_dash='dash', line_color='gray',
                annotation_text='Random baseline (AUC=0.5)'
            )
            fig_comp.update_layout(
                barmode='group',
                title='Link Prediction AUC: Random Hold-out vs Temporal Split',
                yaxis_title='AUC-ROC',
                yaxis_range=[0, 1.05],
                height=450,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.image(
            'outputs/figures/temporal_link_pred_comparison.png',
            caption='Temporal vs MUSAE link prediction comparison (pre-computed)',
            use_column_width=True
        )

    st.markdown("---")

    # ── Section 5: Edge Explorer ───────────────────────────────────────────
    st.subheader('5. Temporal Edge Explorer')
    st.caption("Browse the raw temporal edge list — filter by date or domain")

    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.selectbox(
            'Filter by date',
            options=['All'] + sorted(temporal_edges_df['date'].unique().tolist())
        )
    with col2:
        if 'domain' in temporal_edges_df.columns:
            selected_domain = st.selectbox(
                'Filter by domain',
                options=['All', 'ML', 'Web']
            )
        else:
            selected_domain = 'All'

    filtered = temporal_edges_df.copy()
    if selected_date != 'All':
        filtered = filtered[filtered['date'] == selected_date]
    if selected_domain != 'All' and 'domain' in filtered.columns:
        filtered = filtered[filtered['domain'] == selected_domain]

    st.dataframe(
        filtered.head(100),
        use_container_width=True
    )
    st.caption(f"Showing top 100 of {len(filtered):,} matching edges")
