# :maple_leaf: 메이플스토리 코디 추천

<img src="https://user-images.githubusercontent.com/59256704/217677181-3705536e-a046-4a86-bda4-d16e2ce5e627.png" width="800">

## [메신사 체험하기](http://mesinsa.co.kr)


## 1. 프로젝트 소개

<img src="https://user-images.githubusercontent.com/59256704/217676239-130074eb-04dd-40ec-bd90-79807beba883.png" width="800">


메이플스토리 세상에 존재하는 너무나도 많은 아이템들! 현실에서 무슨 옷 입을지 고민하기도 바쁜데 메이플 세계에서도 고민하느라 힘드셨던 용사님들을 위한 코디 추천 서비스입니다. 

고정하고 싶은 아이템을 입력하면 그 아이템과 어울리는 코디 조합 세 개를 추천해드립니다. 이 외에도 용사님들의 코디 점수를 평가 해주는 코디 진단 서비스도 있으니 한 번 방문해서 즐겨보세요!

<br>

<br>

## 2. AI 모델 구조

아이템 임베딩 벡터를 생성하기 위해 이미지 기반 모델을 사용하여 유저와 Interaction이 없는 아이템도 추천 대상에 포함될 수 있도록 했습니다.

- Multi-Layered Comparison Network

<img src="https://user-images.githubusercontent.com/59256704/217676676-b95ca45a-b341-4880-a158-6e4da572e699.png" width="800">

- CNN의 각 Layer에서 서로 다른 Feature를 추출하는 과정을 부위 별로 반복하여 레이어 개수 * 7 만큼의 Feature를 추출합니다.
- 추출한 Feature들 간의 Pairwise 유사성을 파악한 후 MLP Predictor를 통화개 최종 Score를 산출합니다.

<br>

<br>

## 3. 서비스 아키텍쳐

- 데이터 & 모델 학습 파이프라인

<img src="https://user-images.githubusercontent.com/59256704/217677382-78ba2243-c345-46a9-991c-eec247e28e71.png" width="700">

<br>

- 웹 배포 파이프라인

<img src="https://user-images.githubusercontent.com/59256704/217677501-3db7b44c-68f5-48ac-869c-10041a8c00b9.png" width="700">

<br>

- 사용자 요청 흐름도

<img src="https://user-images.githubusercontent.com/59256704/217677580-78d7f68a-7f73-478e-8693-a8429be22607.png" width="700">


<br>

<br>

## 4. 브랜치 전략

<img src="https://user-images.githubusercontent.com/59256704/217677683-5da757bd-d5c9-4f4d-9d72-5d9961cccfc5.png" width=700>

- dev, modeling, back, front 네 개의 브랜치 분류 사용
- 각 브랜치에서 기능 추가, 오류처리 등에 대한 세부 브랜치를 만들어 코드 관리

<br>

<br>

## 5. 사용 기술 스택

<img src="https://user-images.githubusercontent.com/59256704/217677774-8a5b3d48-cd9a-4428-8deb-365ec12ea741.png" width="500">

<br>

<br>

## 6. 팀원 소개

| <img src="https://user-images.githubusercontent.com/64895794/200263288-1d77b5f8-ed79-4548-9bc1-01aec2474aaa.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263509-9f564042-6da7-4410-a820-c8198037b0b3.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263683-37597e1d-10c1-483c-90f2-fb4749310e40.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263783-52ddbcf3-5e0b-431e-a84d-f7f17f3d061e.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200264314-77728a99-9849-41e9-b13d-be120877a184.png" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [류명현](https://github.com/ryubright)                                            |                                           [이수경](https://github.com/41ow1ives)                                            |                                            [김은혜](https://github.com/kimeunh3)                                            |                                         [정준환](https://github.com/Jeong-Junhwan)                                          |                                            [장원준](https://github.com/jwj51720)                                            |



----
🦖 team 공룡알에 대한 더 자세한 WIKI는 [팀 노션](https://41ow1ives.notion.site/Final-Project-7297be317b9340b5a7c18d70c010f783)에서 볼 수 있어요
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fboostcampaitech4lv23recsys1/final-project-level3-recsys-01/2F&count_bg=%23FFE565&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
