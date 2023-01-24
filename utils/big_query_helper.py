from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient
import pandas as pd

from typing import Optional, Sequence

convert_dtype = {"INTEGER": "int64", "STRING": "str"}


class BigQueryHelper:
    """
    BigQuery로 데이터를 올리고 내릴 때 사용하는 객체
    """

    def __init__(self, key_path: str, dataset_name: str = "train_dataset") -> None:
        """
        BigQueryHelper 객체 생성

        Args:
            key_path: 아까 저장한 key file 경로
                ex) keys/gcs_key.json

            dataset: BigQuery 안에 있는 dataset 이름
                ex) train_dataset
        """
        # client 가져오기
        self.bigquery_client = bigquery.Client.from_service_account_json(key_path)
        self.bigquery_read_client = BigQueryReadClient.from_service_account_file(
            key_path
        )
        self.dataset = self.bigquery_client.get_dataset(dataset_name)

    def change_dataset(self, dataset_name: str) -> None:
        """
        현재 선택하고 있는 dataset을 변경

        Args:
            dataset_name(str): 변경 할 dataset의 이름
                ex) train_dataset
        """
        self.dataset = self.bigquery_client.get_dataset(dataset_name)

    def read_df_from_table(self, table_name: str) -> pd.DataFrame:
        """
        선택한 dataset 내에 있는 table을 DataFrame 형태로 로드

        Args:
            table_name (str): 현재 dataset내에 존재하는 table 이름
                ex) newMF

        Returns:
            pd.DataFrame: DataFrame으로 변환된 table
        """
        table_reference = self.dataset.table(table_name)
        return self.bigquery_client.list_rows(table_reference).to_dataframe(
            bqstorage_client=self.bigquery_read_client, progress_bar_type="tqdm"
        )

    def insert_dataframe_to_table(
        self, table_name: str, DataFrame: pd.DataFrame
    ) -> Optional[Sequence[dict]]:
        """
        DataFrame을 table에 업로드. 이 때, table과 DataFrame은 컬럼 이름, 스키마 등이 모두 동일해야한다.
        타입은 억지로 자동변환 될 수 있으므로 유의하자.
            ex) codi-hat, codi_hat은 서로 호환될 수 없는 column 이름

        Args:
            table_name (str):
                DataFrame을 업로드 할 테이블. 반드시 현재 dataset내에 존재해야함.
                다른 dataset으로 올리고 싶다면 `change_dataset` 함수를 활용하여 현재 선택된 dataset을 변경할 것.
                    ex) newMF

            DataFrame (pd.DataFrame): 업로드 할 DataFrame

        Returns:
            Optional[Sequence[dict]]: 오류가 있으면, 첫번째 발생 오류를 return, 없으면 None이다.
        """
        table_reference = self.dataset.table(table_name)
        table = self.bigquery_client.get_table(table_reference)
        schemas = table.schema

        if len(schemas) != DataFrame.shape[1]:
            raise ValueError(
                f"Table의 column 개수 {len(schemas)}와 DataFrame의 column 개수 {DataFrame.shape[1]}가 달라요."
            )

        for i in range(len(schemas)):
            schema = schemas[i]
            col = DataFrame.columns[i]

            if schema.name != col:
                raise ValueError(
                    f"column 이름이 일치하지 않습니다. Table: {schema.name}, DataFrmae: {col}"
                )

            DataFrame[col] = DataFrame[col].astype(convert_dtype[schema.field_type])

        errors = self.bigquery_client.insert_rows_from_dataframe(
            table=table, dataframe=DataFrame
        )

        for error in errors:
            if error:
                return error

        return None
