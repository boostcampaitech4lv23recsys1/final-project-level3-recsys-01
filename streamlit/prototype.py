import streamlit as st
import pandas as pd
from modeling.model.newMF import NewMF
import modeling.inference import main


# import data
item = pd.read_csv('./data/new_item.csv')
# import model
model = NewMF
model.load_state_dict(torch.load('./model/NewMF.pt'))
model.eval()

st.text_input('config_value')