import streamlit as st, runpy
import ui_theme as ui

st.set_page_config(page_title="UBR · 시방서 QA 챗봇", layout="wide")
ui.apply()
ui.hero("시방서 QA 챗봇", "문서 질의응답")

runpy.run_path("chatbot.py", run_name="__main__")
