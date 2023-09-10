import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# Welcome! 👋")

st.sidebar.success("Select pages above.")

st.markdown(
    """
    👈 Необходимо выбрать одну из вкладок в левой части экрана
    
    **Overview** - общий обзор таргетов
    
    **Thresholds** - подробный обзор таргетов
"""
)