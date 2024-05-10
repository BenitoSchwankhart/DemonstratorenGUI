import streamlit as st

# Sidebar-Konfiguration 
st.set_page_config(
    page_title="Meine Multipage App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar-Navigation
pages = {
    "Home": "home",
    "Seite 1": "page1",
    "Seite 2": "page2"
}

selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# Laden der ausgewählten Seite 
page = __import__(f"pages.{pages[selected_page]}")
page.main()