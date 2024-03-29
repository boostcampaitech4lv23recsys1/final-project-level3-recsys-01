from google.cloud import storage
from torch.nn import Module
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm


class GCSHelper:
    """
    GCS로 파일을 주고 받을 때 사용하는 객체
    """

    def __init__(
        self, key_path: str, bucket_name: str = "maple_preprocessed_data"
    ) -> None:
        """
        key_path: 아까 저장한 key file 경로 ex) config/key.json
        bucket_name: gcs 안에 어떤 버킷에 저장 할 것인지? ex) maple_raw_data
        """
        # client 가져오기
        self.storage_client = storage.Client.from_service_account_json(key_path)
        # client 내 bucket 가져오기
        self.bucket = self.storage_client.get_bucket(bucket_name)

    def change_bucket(self, bucket_name: str):
        """
        bucket_name: 현재 설정되어있는 버킷 변경 ex) maple_preprocessed_data
        """
        self.bucket = self.storage_client.get_bucket(bucket_name)

    def upload_file_to_gcs(self, blob_name: str, file_name: str) -> None:
        """
        파일 하나를 gcs로 업로드

        blob_name: gcs에 저장될 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
        file_name: gcs에 upload할 파일 경로 및 이름 (local 상의 경로) ex) data/user_info/user_info.csv
        """
        # bucket 내 깡통 blob 생성
        blob = self.bucket.blob(blob_name)
        # 해당 blob에 파일 업로드
        blob.upload_from_filename(file_name)
        return None

    def download_file_from_gcs(self, blob_name: str, file_name: str) -> None:
        """
        파일을 gcs로 부터 다운로드

        blob_name: gcs에 저장되어있는 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
        file_name: gcs에서 download할 파일 경로 및 이름 (local 상의 경로) ex) data/user_info/user_info.csv
        """
        # bucket 내 blob 가져오기
        blob = self.bucket.blob(blob_name)
        # 해당 blob에 파일 업로드
        blob.download_to_filename(file_name)
        return None

    def download_folder_from_gcs(self, folder_name: str, save_path: str) -> None:
        """
        `download_file_from_gcs`는 파일 하나를 다운로드.
        얘는 폴더를 입력받아 폴더 전체를 다운로드.
        폴더 안에 폴더가 있을 경우는 구현하지 않음

        Args:
            folder_name: gcs bucket 내에 존재하는 폴더 이름 ex) image/item
            save_path: bucket에 존재하는 파일을 저장할 폴더 이름 ex) modeling/data/image/item
        """

        # 저장할 폴더 없으면 생성
        os.makedirs(save_path, exist_ok=True)
        # 해당 폴더 내에 있는 모든 blob 가져오기
        blobs = self.bucket.list_blobs(prefix=folder_name)
        # 블롭에 대해 하나씩 다운로드 실행
        for blob in tqdm(blobs):
            if blob.name.endswith("/"):
                continue
            file_name = blob.name.split("/")[-1]
            save_file_path = os.path.join(save_path, file_name)
            self.download_file_from_gcs(blob.name, save_file_path)
        return None

    def open_image_from_gcs(self, blob_name: str) -> Image:
        """
        image를 `PIL.Image` 형식으로 return

        Args:
            blob_name: image file 이름 ex) image/item/123456.png

        Returns:
            PIL.Image: 이미지...
        """

        blob = self.bucket.get_blob(blob_name)
        memory_file = BytesIO()
        blob.download_to_file(memory_file)
        image = Image.open(memory_file).convert("RGB")
        return image

    def upload_df_to_gcs(self, blob_name: str, df: pd.DataFrame) -> None:
        """
        DataFrame을 gcs로 바로 업로드

        blob_name: gcs에 저장될 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
        df: gcs에 upload할 DataFrame
        """

        # stream을 이용해 바로 업로드
        with self.bucket.blob(blob_name).open("w") as f:
            df.to_csv(f, encoding="utf-8-sig", index=False)
        return None

    def read_df_from_gcs(self, blob_name: str) -> pd.DataFrame:
        """
        gcs에 있는 csv를 바로 pandas로 read

        blob_name: gcs에 저장된 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
        """

        # stream을 이용해 바로 업로드
        with self.bucket.blob(blob_name).open("r") as f:
            df = pd.read_csv(f, encoding="utf-8-sig")
        return df

    def upload_image_to_gcs(self, blob_name: str, image_url: str) -> None:
        """
        url 안의 이미지를 gcs로 바로 업로드

        blob_name: gcs에 저장될 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
        image_url: gcs에 upload할 image의 url
        """
        # 이미지 읽어오기
        image_data = requests.get(image_url).content
        # stream을 이용해 바로 업로드
        with self.bucket.blob(blob_name).open("wb") as f:
            f.write(image_data)
        return None

    def upload_model_to_gcs(self, blob_name: str, model_name: str) -> None:
        """
        model.pt를 gcs로 바로 업로드

        blob_name: gcs에 저장될 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
        model_name: gcs에 upload할 model 경로 및 이름 (local 상의 경로) ex) data/user_info/user_info.csv
        """
        # bucket 내 깡통 blob 생성
        blob = self.bucket.blob(blob_name)
        # 해당 blob에 파일 업로드
        blob.upload_from_filename(model_name)
        return None

    def path_exists(self, path: str):
        return storage.Blob(bucket=self.bucket, name=path).exists(self.storage_client)


# 일단 주석 처리
# def image_upload(
#     blob_name: str,
#     image_url: str,
#     key_path: str = "key.json",
#     bucket_name: str = "maple_raw_data",
# ):
#     """
#     url 안의 이미지를 gcs로 바로 업로드

#     blob_name: gcs에 저장될 파일 경로 및 이름 (gcs 상의 경로) ex) csv/user_info.csv
#     image_url: gcs에 upload할 image의 url
#     key_path: 아까 저장한 key file 경로 ex) config/key.json
#     bucket_name: gcs 안에 어떤 버킷에 저장 할 것인지? ex) maple_raw_data
#     """
#     # 이미지 읽어오기
#     image_data = requests.get(image_url).content
#     # client 가져오기
#     storage_client = storage.Client.from_service_account_json(key_path)
#     # client 내 bucket 가져오기
#     bucket = storage_client.get_bucket(bucket_name)
#     # stream을 이용해 바로 업로드
#     with bucket.blob(blob_name).open("wb") as f:
#         f.write(image_data)
#     return None


def add_gcs_image_path(user_detail_info: pd.DataFrame) -> pd.DataFrame:
    """
    이미지를 gcs에 저장하면 user 정보에서 image를 gcs에서 가져올 수 있도록 바꿔야한다.
    gcs 내 주소를 언급해주어 접근 경로를 web이 아니라 우리의 클라우드가 되도록 하자.

    cur_chr, past_chr_img_1 ~ 6: 현재 image 정보를 가지고 있는 column (web에 존재)
    뒤에 _gcs를 붙인 새로운 컬럼 7개를 만들어준다. ex) cur_chr_gcs, past_chr_img_1_gcs

    cur_chr의 경우 날짜 정보가 없기때문에 유저가 직접 입력해줘야한다.
    "221223" 형식의 str을 이용하자.
    """

    cur_date = sorted(user_detail_info["last_access"], reverse=True)[0].replace(
        "/", "_"
    )

    # {경로}_{닉네임}_{저장날짜} 형식
    user_detail_info[
        "gcs_image_path"
    ] = f"image/user/{user_detail_info['nickname']}/{user_detail_info['nickname']}_{cur_date}"
    return user_detail_info
