import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modeling.model.newMF import NewMF
from modeling.inference import main
from modeling.utilities import read_json, loading_text_file
import torch
import json


# import data
item = pd.read_csv('./data/new_item.csv')
inference_start = False
################################################################

st.title("공룡알 prototype")
st.write("NewMF를 이용한 공룡알의 최종 프로젝트 프로토타입입니다.")
st.write("먼저 사이드 바에서 추천 받고 싶은 부위와 고정하고자 하는 아이템들을 입력해주세요.")

################################################### Sidebar
codi_list = ['모자', '헤어', '성형','상의', '하의','신발', '무기']
st.sidebar.markdown("### 1. 추천 받고 싶은 부위를 선택합니다.")
rec_buwi = st.sidebar.multiselect("추천 받고 싶은 부위를 선택해주세요.", options=codi_list)

st.sidebar.markdown("### 2. 고정하고자 하는 아이템을 선택합니다.")
hat_option, hair_option, face_option, top_option, bottom_option, shoes_option, weapon_option = ['-']*7
codi_option = [hat_option, hair_option, face_option, top_option, bottom_option, shoes_option, weapon_option]
if '모자' not in rec_buwi:
    hat_option = st.sidebar.selectbox(
        '고정하실 모자를 선택해주세요',
        item[item['subCategory']=='Hat']['name'].tolist()
    )
if '헤어' not in rec_buwi:
    hair_option = st.sidebar.selectbox(
        '고정하실 헤어를 선택해주세요.',
        item[item['subCategory']=='Hair']['name'].tolist()
    )
if '성형' not in rec_buwi:
    face_option = st.sidebar.selectbox(
        '고정하실 성형을 선택해주세요.',
        item[item['subCategory']=='Face']['name'].tolist()
    )
if '상의' not in rec_buwi:
    top_option = st.sidebar.selectbox(
        '고정하실 상의를 선택해주세요.',
        item[item['subCategory']=='Top']['name'].tolist()
    )
if '하의' not in rec_buwi:
    bottom_option = st.sidebar.selectbox(
        '고정하실 하의를 선택해주세요.',
        item[item['subCategory']=='Bottom']['name'].tolist()
    )
if '신발' not in rec_buwi:
    shoes_option = st.sidebar.selectbox(
        '고정하실 신발을 선택해주세요.',
        item[item['subCategory']=='Shoes']['name'].tolist()
    )
if '무기' not in rec_buwi:
    weapon_option = st.sidebar.selectbox(
        '고정하실 무기를 선택해주세요.',
        item[item['subCategory'].str.find('Weapon')!=-1]['name'].tolist()
    )
    
if st.sidebar.button("추천 시작!"):
    # import model
    config = read_json("./modeling/config/mfconfig.json")
    cur_codi = {"codi-hat":hat_option,
                            "codi-hair":hair_option,
                            "codi-face":face_option,
                            "codi-top":top_option,
                            "codi-bottom":bottom_option,
                            "codi-shoes":shoes_option,
                            "codi-weapon":weapon_option}
    codi_index = {}
    for codi, key in zip(codi_option, cur_codi.keys()):
        if codi == '-':
            codi_index[key] = -1
        else:
            try: #key가 안 맞아서 일단 안 되면 -1 뱉도록 함
                codi_index[key] = item2idx[codi]
            except:
                codi_index[key] = -1
                
    equipment = {"cur_codi":cur_codi, "fix_equip" : codi_index}

    inference = main(config, equipment)
    st.dataframe(inference)
