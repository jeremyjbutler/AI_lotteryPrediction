import streamlit as st
from dataframes import LottoDataFrame
import pandas as pd

from load_css import local_css

local_css("style.css")

lotto = LottoDataFrame()

powerballdata = lotto.get_powerball()

btext="Baysian Model"
#column_names = ['b1', 'b2', 'b3', 'b4', 'b5', 'pb']
#dfchart = pd.DataFrame(dfindex, lotto.get_powerball(), columns=column_names)

st.title("AI Lottery Predictions")



def on_button_click():
    result_string = " "
    initstring="<div id=\"pball\">"
    #st.text(btext)
    for value in powerballdata.values():
        result_string += "<div id=\"pball\"> " + str(value) + "     </span>    "
        #st.text(value)
    st.markdown(f"{btext} {initstring} {result_string}", unsafe_allow_html=True)
    #"Baysian Model", result_string
    #st.text(res)


# Create the button
if st.button("Get Numbers"):
    on_button_click()
    
#final_table = st.table(dfchart)
#final_table.add_rows(powerballdata)
#st.table(powerballdata)