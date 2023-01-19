## Back-end README

Dockerfile은 아직 미완성입니다...

가상환경 하나 만드시고 requirements.txt 설치해주세요.

gcs_key.json를 keys/gcs_key.json 경로에 넣고 경로 설정을 src.database.database_creation.py 에다가 해주세요.
이거를 어떻게 깔끔하게 하는 방법이 있는지는 저도 조금 더 고민해보겠습니다.
utils에도 보면 gcs helper가 또 있어요.
근데 backend는 이 폴더만 따로 VM에 올라갈거라 밖에 있는걸 가져와서 쓰면 안될 것 같았어요.

먼저 mysql이 설치되어 있어야 합니다.
설치 후에, mysql 서버를 열어주세요.
$ service mysql start
$ sudo /etc/init.d/mysql start
뭐가 맞는건지 아직 잘 모르겠습니다 ㅈㅅ

그다음 `python3 -m src.main` 하셔서 실행해주시면 됩니다.
