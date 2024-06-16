import streamlit as st
from dataframes import LottoDataFrame
import pandas as pd
import urllib3

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
    for value in list(powerballdata.values())[0:6]:
        result_string += initstring + str(value) + " </div> "
        #st.text(value)
        powerballresult = list(powerballdata.values())[-1]
    st.markdown(f"{initstring} {result_string} </span>", unsafe_allow_html=True)
   
    #"Baysian Model", result_string
    #st.text(res)

def on_update_click():
    #download results
    pass
    
# Create the button
if st.button("Get Numbers"):
    on_button_click()
if st.button("Update Past Results"):
    on_update_click()
    
#final_table = st.table(dfchart)
#final_table.add_rows(powerballdata)
#st.table(powerballdata)