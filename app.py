import streamlit as st
from dataframes import LottoDataFrame


lotto = LottoDataFrame()

powerballdata = lotto.get_powerball()

st.title("AI Lottery Predictions")
st.text