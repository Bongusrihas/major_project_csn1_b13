"""Focused navigation for the two-method research project."""
from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.dashboard import apply_style, watch_latest_run

st.set_page_config(page_title='Disease trajectories ', page_icon='◈', layout='wide')
apply_style()
watch_latest_run()
page = st.navigation([
    st.Page('pages/1_Published_paper.py', title='Paper vs proposed', icon=':material/article:', default=True),
    st.Page('pages/2_Our_improvement.py', title='Proposed framework', icon=':material/compare_arrows:'),
    st.Page('pages/3_3D_explorer.py', title='3D explorer', icon=':material/deployed_code:'),
    st.Page('pages/4_Random_dataset_tests.py', title='Dataset tests', url_path='Random_dataset_tests', icon=':material/science:'),
])
# st.sidebar.markdown('**Two methods**\n\nPaper framework\n\nProposed framework')
# st.sidebar.caption('Prepared artifacts are read from disk. Changing a view never starts training.')
page.run()
