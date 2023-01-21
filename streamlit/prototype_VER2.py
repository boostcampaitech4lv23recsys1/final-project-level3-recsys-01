import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modeling.model.newMF import NewMF
from modeling.inference import main
from modeling.utilities import read_json, loading_text_file
from utils import GCSHelper
import torch
import json
import re

def __preprocess(set_buwi: str) -> str:
    pattern = r"[!@#$%^&*()_+=`~,.<>/?{}\s\[\]\-:]"
    set_buwi = re.sub(pattern, "", set_buwi.lower())
    return set_buwi

# import data
gcs_helper = GCSHelper(key_path = "./keys/gcs_key.json", bucket_name = "maple_preprocessed_data")
item = gcs_helper.download_file_from_gcs(blob_name = "item_KMST_1149_VER1.2.csv", file_name = "./data/item_KMST_1149_VER1.2.csv")
item = pd.read_csv('./data/item_KMST_1149_VER1.2.csv')
inference_start = False
################################################################

'''
# 공룡알 prototype
NewMF를 이용한 공룡알의 최종 프로젝트 프로토타입입니다. \n
'''
st.info("#### 1️⃣ 추천 받고 싶은 부위와 고정하고자 하는 아이템들을 입력해주세요.")
################################################### Sidebar
codi_list = ['모자', '헤어', '성형','상의', '하의','신발', '무기']
st.markdown("##### 1. 추천 받고 싶은 부위를 선택합니다.")
rec_buwi = st.multiselect("추천 받고 싶은 부위를 선택해주세요.", options=codi_list)

st.markdown("##### 2. 고정하고자 하는 아이템을 선택합니다.")
hat_option, hair_option, face_option, top_option, bottom_option, shoes_option, weapon_option = ['-']*7
if '모자' not in rec_buwi:
    hat_option = st.selectbox(
        '고정하실 모자를 선택해주세요',
        item[item['subCategory']=='Hat']['name'].tolist()
    )
    hat_option = __preprocess(hat_option)
if '헤어' not in rec_buwi:
    hair_option = st.selectbox(
        '고정하실 헤어를 선택해주세요.',
        item[item['subCategory']=='Hair']['name'].tolist()
    )
    hair_option = __preprocess(hair_option)
if '성형' not in rec_buwi:
    face_option = st.selectbox(
        '고정하실 성형을 선택해주세요.',
        item[item['subCategory']=='Face']['name'].tolist()
    )
    face_option = __preprocess(face_option)
if '상의' not in rec_buwi:
    top_option = st.selectbox(
        '고정하실 상의를 선택해주세요.',
        item[item['subCategory']=='Top']['name'].tolist()
    )
    top_option = __preprocess(top_option)
if '하의' not in rec_buwi:
    bottom_option = st.selectbox(
        '고정하실 하의를 선택해주세요.',
        item[item['subCategory']=='Bottom']['name'].tolist()
    )
    bottom_option = __preprocess(bottom_option)
if '신발' not in rec_buwi:
    shoes_option = st.selectbox(
        '고정하실 신발을 선택해주세요.',
        item[item['subCategory']=='Shoes']['name'].tolist()
    )
    shoes_option = __preprocess(shoes_option)
if '무기' not in rec_buwi:
    weapon_option = st.selectbox(
        '고정하실 무기를 선택해주세요.',
        item[item['subCategory'].str.find('Weapon')!=-1]['name'].tolist()
    )
    weapon_option = __preprocess(weapon_option)
    
if st.button("추천 시작!"):
    # import model
    st.info("#### 2️⃣ 추천 결과는 아래와 같습니다.")
    config = read_json("./modeling/config/mfconfig.json")
    cur_codi = {"codi-hat":hat_option,
                "codi-hair":hair_option,
                "codi-face":face_option,
                "codi-top":top_option,
                "codi-bottom":bottom_option,
                "codi-shoes":shoes_option,
                "codi-weapon":weapon_option}
    codi_index = {}
    print(cur_codi.items())
    for idx, (key, codi) in enumerate(cur_codi.items()):
        if codi != '-':
            codi_index[key] = -1
        else:
            try: #key가 안 맞아서 일단 안 되면 -1 뱉도록 함
                codi_index[key] = idx
            except:
                codi_index[key] = -1
                
    equipment = {"cur_codi":cur_codi, "fix_equip" : codi_index}

    inference = main(config, equipment)
    new_column_name = [ f"BEST{i}" for i in range(1,len(inference.columns)+1)]
    inference.columns = new_column_name
    inference = inference[[ f"BEST{i}" for i in range(1,6)]]
    inference.index = cur_codi.keys()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([f"BEST{i}" for i in range(1,6)])
    tab_lists = [tab1, tab2, tab3, tab4, tab5]
    for idx, tab in enumerate(tab_lists):
        with tab:
            cur_recommendation = inference[inference.columns[idx]]
            hat, hair, face, top, bottom, shoes, weapon = st.columns(7)
            tab_codi_columns = [hat, hair, face, top, bottom, shoes, weapon]
            for tab_codi, codi_name in zip(tab_codi_columns, cur_codi.keys()):
                with tab_codi:
                    st.warning(cur_recommendation[codi_name])
                    
