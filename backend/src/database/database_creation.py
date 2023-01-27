import pymysql
import pandas as pd

from src.utils import GCSHelper


def create_db():
    # 실행중인 db server에 접속
    host_name = "localhost"
    port = 3306
    user_name = "root"

    db = pymysql.connect(host=host_name, port=port, user=user_name, charset="utf8")

    # items라는 이름의 DATABASE 만들고, 선택
    # 혹시 이미 존재하면 걍 지워버리고 새로 만들기
    cursor = db.cursor()
    cursor.execute("DROP DATABASE IF EXISTS items;")
    cursor.execute("CREATE DATABASE items;")
    db.select_db("items")

    # items라는 DATABASE안에 backend용 TABLE 생성
    sql_cmd = """
    CREATE TABLE backend(
        id INT NOT NULL AUTO_INCREMENT,
        requiredGender INT,
        isCash BOOLEAN,
        name VARCHAR(60),
        category VARCHAR(40),
        subCategory VARCHAR(60),
        gcsImageUrl VARCHAR(60),
        nameProcessed VARCHAR(40),
        equipCategory VARCHAR(10),
        PRIMARY KEY(ID)
    );
    """
    cursor.execute(sql_cmd)

    # 위의 TABLE에 집어 넣을 csv 파일 불러오기
    key_path = "keys/gcs_key.json"
    bucket_name = "maple_preprocessed_data"
    gcs_helper = GCSHelper(key_path=key_path, bucket_name=bucket_name)

    file_name = "item_KMST_1149_VER1.2.csv"
    item_info = gcs_helper.read_df_from_gcs(file_name)

    # item_info = pd.read_csv("../item_KMST_1149_VER1.2.csv", encoding="utf-8-sig")
    columns = [
        "requiredGender",
        "isCash",
        "name",
        "category",
        "subCategory",
        "gcs_image_url",
        "name_processed",
        "equipCategory",
    ]
    item_info = item_info[columns]
    item_info.reset_index(names="id", inplace=True)
    item_info["id"] += 1

    # 가져온 데이터를 하나씩 DB에 집어넣기.
    sql_cmd = "INSERT INTO backend(id, requiredGender, isCash, name, category, subCategory, gcsImageUrl, nameProcessed, equipCategory) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    for _, row in item_info.iterrows():
        row = list(row)
        cursor.execute(sql_cmd, row)

    # 종료!
    db.commit()
    db.close()
