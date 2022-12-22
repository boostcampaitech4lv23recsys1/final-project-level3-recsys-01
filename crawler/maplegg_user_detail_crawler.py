import requests
from bs4 import BeautifulSoup
from itertools import chain
from tqdm import tqdm
import datetime
import time
import argparse
import pandas as pd
import os

COLUMNS = ['nickname',
            'codi-hat',
            'codi-hair',
            'codi-face',
            'codi-top',
            'codi-bottom',
            'codi-shoes',
            'codi-weapon',
            'level',
            'class',
            'popularity',
            'total_ranking',
            'world_ranking',
            'class_world_ranking',
            'class_total_ranking',
            'guild',
            'last_access',
            'mureung',
            'theseed',
            'union',
            'achievement',
            'cur_chr',
            'past_chr_img_1',
            'past_chr_img_2',
            'past_chr_img_3',
            'past_chr_img_4',
            'past_chr_img_5',
            'past_chr_img_6',
            'past_chr_date_1',
            'past_chr_date_2',
            'past_chr_date_3',
            'past_chr_date_4',
            'past_chr_date_5',
            'past_chr_date_6']


def get_codi_analysis(soup):
    codi = soup.find_all('span', attrs={'class':'character-coord__item-name'})
    codi_list = [cd.text for cd in codi]
    # ['기억의 갈래', '숲의 요정', '-', '호수 물결', '-', '신기루 자욱', '망각의 등불']
    return codi_list

def get_chr_info(soup):
    chr = soup.find_all('li', attrs={'class':'user-summary-item'})
    chr_list = [ info.text for info in chr]
    # ['Lv.289(28.778%)', '엔젤릭버스터', '인기도\n12,239']
    return chr_list

def get_guild_ranking(soup):
    guild = soup.find('div', attrs={'class':'col-lg-2 col-md-4 col-sm-4 col-12 mt-3'})
    guild = guild.text[4:].replace('\n','')
    ranking = soup.find_all('div', attrs={'class' : 'col-lg-2 col-md-4 col-sm-4 col-6 mt-3'})
    ranking_list = [ rank.text[5:].replace('\n','').replace(' ','') for rank in ranking]
    ranking_list.append(guild)
    # ['3위', '2위', '(월드)1위', '(전체)1위', '드루']
    return ranking_list

def get_last_access(soup):
    last = soup.find('div', attrs={'class':'col-6 col-md-8 col-lg-6'})

    if (last == '') or (type(last) is None):
        last_visit = '-'
    else:
        last = last.text.replace('\n', '')
        last = int(last.replace(' ','')[7:-2])
        last_visit = (datetime.datetime.now() - datetime.timedelta(days=last)).strftime("%y/%m/%d")
    # '22/12/18'
    return [last_visit]

def get_mureung_theseed_union_achieve(soup):
    datas = soup.findAll("div", "col-lg-3 col-6 mt-3 px-1")
    for idx, data in enumerate(datas):
        if idx==0:
            mureung = data.find('h1')
            if mureung is None:
                mureung = "기록이 없습니다."
            else:
                mureung = mureung.text.split('\n')[0]
        elif idx==1:
            theseed = data.find('h1')
            if theseed is None:
                theseed = "기록이 없습니다."
            else:
                theseed = theseed.text.split('\n')[0]
        elif idx==2:
            union = data.find('span')
            if union is None:
                union = "기록이 없습니다."
            else:
                union = union.text[3:]
        elif idx==3:
            achieve = data.find('span')
            if achieve is None:
                achieve = "기록이 없습니다."
            else:
                achieve = achieve.text[5:]
    return [mureung, theseed, union, achieve]

def get_cur_chr_img(soup):
    now_chr = soup.find('div', attrs={'class':'col-6 col-md-8 col-lg-6'}).find('img')['src']
    return [now_chr]

def get_past_chr_img_date(soup):
    # 과거 캐릭터 이미지와 해당 날짜
    past_chr = soup.find_all('div', attrs={'class': 'avatar-collection-item col-lg-2 col-md-4 col-6'})
    past_chr_img_list = [ past_chr_info.find('img')['src'] for past_chr_info in past_chr]
    if len(past_chr_img_list)!=6:
        past_chr_img_list.extend(['None'] * (6-len(past_chr_img_list)))
    past_chr_day_list = [ int(past_chr_info.find('img')['alt'].split('(')[1][:-4]) for past_chr_info in past_chr]
    past_chr_day_list = [ (datetime.datetime.now() - datetime.timedelta(days=last)).strftime("%y/%m/%d") for last in past_chr_day_list]
    if len(past_chr_day_list)!=6:
        past_chr_day_list.extend(['None'] * (6-len(past_chr_day_list)))
    past_chr_img_list.extend(past_chr_day_list)
    return past_chr_img_list

def crawler(url):
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    try:
        codi_list = get_codi_analysis(soup)
        chr_info = get_chr_info(soup)
        guild_ranking = get_guild_ranking(soup)
        last_access = get_last_access(soup)
        mureung_theseed_union_achieve = get_mureung_theseed_union_achieve(soup)
        cur_chr = get_cur_chr_img(soup)
        past_chr = get_past_chr_img_date(soup)
    except:
        req = requests.get(url)
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')
        codi_list = get_codi_analysis(soup)
        chr_info = get_chr_info(soup)
        guild_ranking = get_guild_ranking(soup)
        last_access = get_last_access(soup)
        mureung_theseed_union_achieve = get_mureung_theseed_union_achieve(soup)
        cur_chr = get_cur_chr_img(soup)
        past_chr = get_past_chr_img_date(soup)
    final_list = list(chain(codi_list,
                            chr_info,
                            guild_ranking,
                            last_access,
                            mureung_theseed_union_achieve,
                            cur_chr,
                            past_chr))
    return final_list



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="유저의 정보를 크롤링합니다.")
    parser.add_argument("--url_dir", required=True, help="csv가 있는 디렉토리를 입력해주세요.")
    parser.add_argument("--save_dir", required=True, help="파일을 저장할 디렉토리를 입력해주세요.")
    parser.add_argument("--save_every", default=1000)
    args = parser.parse_args()

    user_num_list = os.listdir(args.url_dir)
    
    for user_num in user_num_list:
        data = pd.read_csv(os.path.join(args.url_dir, user_num))
        user_num = user_num.split('user_info_')[1]
        print(f"----------{user_num}----------")
        final_list = []
        for idx, row in enumerate(tqdm(data.values)):
            user = [row[2]]
            user.extend(crawler(row[1]))
            final_list.append(user)
            
            if idx % args.save_every == 0 and not idx==0:
                final_df = pd.DataFrame(final_list, columns=COLUMNS)
                final_df.to_csv(f'{args.save_dir}/user_detail_{user_num}', index=False, encoding='utf-8-sig')
        final_df = pd.DataFrame(final_list, columns=COLUMNS)
        final_df.to_csv(f'{args.save_dir}/user_detail_{user_num}', index=False, encoding='utf-8-sig')
    