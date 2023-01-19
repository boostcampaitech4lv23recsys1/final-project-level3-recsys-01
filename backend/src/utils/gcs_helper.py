from google.cloud import storage
import pandas as pd
import requests


class GCSHelper:
    """
    GCS로 파일을 주고 받을 때 사용하는 객체
    """

    def __init__(self, key_path: str, bucket_name: str = "maple_raw_data") -> None:
        """
        key_path: 아까 저장한 key file 경로 ex) config/key.json
        bucket_name: gcs 안에 어떤 버킷에 저장 할 것인지? ex) maple_raw_data
        """
        # client 가져오기
        self.storage_client = storage.Client.from_service_account_json(key_path)
        # client 내 bucket 가져오기
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

    def path_exists(self, path: str):
        return storage.Blob(bucket=self.bucket, name=path).exists(self.storage_client)
