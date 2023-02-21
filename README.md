<div align="center">
<img src = "https://user-images.githubusercontent.com/78770033/217690517-1fd25740-8b01-41e2-85c0-452d1fd025ce.png" >

 <h1> MESINSA :maple_leaf: 메이플스토리 코디 추천 </h1>
 <br>
 <p> 메이플스토리 세상에 존재하는 너무나도 많은 아이템들! <br />
 현실에서 무슨 옷 입을지 고민하기도 바쁜데 <br />
 메이플 세계에서도 고민하느라 힘드셨던 용사님들을 위한 <b>AI 코디 추천 서비스</b>입니다. <br />
 고정하고 싶은 아이템을 입력하면 그 아이템과 어울리는 코디 조합 세 개를 AI가 추천해드립니다. <br />
 이 외에도 용사님들의 코디 점수를 평가 해주는 코디 진단 서비스도 있으니 한 번 방문해서 즐겨보세요! :maple_leaf: ⚔️<br />
<br>

---
#### <a href="http://mesinsa.co.kr"> 👉 메신사 체험하기 <a/>
---
<a href="https://41ow1ives.notion.site/7297be317b9340b5a7c18d70c010f783"> 팀 노션 <a/> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://drive.google.com/file/d/1fP7xUzF1oJofdpTOMUZ_LWpGZqkBWLh3/view?usp=sharing"> 발표 자료 <a/>&nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://www.youtube.com/watch?v=H1UhgLhY-Ww"> 발표 영상
<br> <br>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fboostcampaitech4lv23recsys1/final-project-level3-recsys-01/2F&count_bg=%23E99533&title_bg=%23555555&title=hits&edge_flat=false"/></a>
<img src="https://img.shields.io/badge/release-1.0.0-E99533"> 
</div>
<br>

## 👚 서비스 구성 요소
### 🌹 코디 추천 받기
<img src = "https://user-images.githubusercontent.com/78770033/217699435-f6315253-0bf9-4f7d-a177-a58626bc4385.png" width=600>

> 1. **고정하고 싶은 부위의 아이템을 선택해요** 
> 2. **코디 추천 받기 버튼을 클릭해요** <br>
100만 명 이상의 코디 조합을 학습한 AI가 선택한 아이템과 잘 어울리는 코디 조합을 추천해드려요.
> 3. **추천 받은 코디셋에 대한 피드백을 남겨주세요** <br>
남겨주신 피드백은 추후 더 좋은 AI가 학습되는데 도움이 됩니다.

### 💯 코디 진단 받기
<img src = "https://user-images.githubusercontent.com/78770033/217701505-7e554351-7210-489e-8b26-489fd96e205d.png" width=600>

> 1. **진단 받고 싶은 코디 조합을 입력해요**
> 2. **코디 진단 받기 버튼을 클릭해요**
> 3. **AI가 코디 점수를 반환해줘요**


<br>
<h2> 💻 활용 장비 및 협업 툴</h2>

- GPU: V100 5대
- 운영체제: Ubuntu 18.04.5 LTS
- 협업툴: Github, Notion, Weight & Bias

<br>

## 🛸 최종 선정 AI 모델 구조

메이플 세계의 다양한 아이템들을 추천하기 위해 이미지 기반 모델을 사용했습니다. <br />
이를 통해 유저와 Interaction이 없는 아이템도 추천 대상에 포함될 수 있도록 했습니다.

#### <b> Multi-Layered Comparison Network<sup>[1]</sup> </b>

<img src="https://user-images.githubusercontent.com/59256704/217676676-b95ca45a-b341-4880-a158-6e4da572e699.png" width="800">
<img src="https://user-images.githubusercontent.com/78770033/217743106-018f1a90-dfd7-46a2-ace2-342eedc41bb1.png" width="800">

- CNN의 각 Layer에서 서로 다른 Feature를 추출하는 과정을 부위 별로 반복하여 레이어 개수 * 7 만큼의 Feature를 추출합니다.
- 추출한 Feature들 간의 Pairwise 유사성을 파악한 후 MLP Predictor를 통과해 최종 Score를 산출합니다.

<br>

## 🚀 서비스 아키텍쳐

- 데이터 & 모델 학습 파이프라인

<img src="https://user-images.githubusercontent.com/59256704/217677382-78ba2243-c345-46a9-991c-eec247e28e71.png" width="700">

<br>

- 웹 배포 파이프라인

<img src="https://user-images.githubusercontent.com/59256704/217677501-3db7b44c-68f5-48ac-869c-10041a8c00b9.png" width="700">

<br>

- 사용자 요청 흐름도

<img src="https://user-images.githubusercontent.com/59256704/217677580-78d7f68a-7f73-478e-8693-a8429be22607.png" width="700">

<br>

## 🦾 사용 기술 스택

<img src="https://user-images.githubusercontent.com/59256704/217677774-8a5b3d48-cd9a-4428-8deb-365ec12ea741.png" width="500">

<br>

## 🦖 팀원 소개

![Untitled](https://user-images.githubusercontent.com/59808674/218402519-4b8b4538-52c6-42d8-a116-51eb5eb3d385.png)


<br>

## 🖇 Appendix
[1] Wang, Xin. Wu, Bo, Zhong, Yueqi. (2019). Outfit compatibility prediction and diagnosis with multi-layered comparison network. MM ‘19 : Proceedings of the 27th ACM International Conference on Multimedia. 27, 329-337.

----
🦖 team 공룡알에 대한 더 자세한 WIKI는 [팀 노션](https://41ow1ives.notion.site/Final-Project-7297be317b9340b5a7c18d70c010f783)에서 볼 수 있어요

