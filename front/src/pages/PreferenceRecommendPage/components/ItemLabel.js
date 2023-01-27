const hat = [
  { label: "파란색 털모자" },
  { label: "흑복건" },
  { label: "청복건" },
  { label: "귀신 가면" },
  { label: "교복모자" },
  { label: "남자 닌자모자" },
  { label: "마개(일본 상투)" },
  { label: "저승사자 갓" },
  { label: "중절모" },
  { label: "메소레인저 레드 헬멧" },
  { label: "메소레인저 블루 헬멧" },
  { label: "메소레인저 그린 헬멧" },
  { label: "크라운 옐로우" },
  { label: "크라운 그린" },
  { label: "크라운 블루" },
  { label: "크라운 레드" },
  { label: "반담 모자" },
];

const hair = [
  { label: "검은색 바람이 분다 헤어" },
  { label: "갈색 바람이 분다 헤어" },
  { label: "검은색 별사탕 헤어" },
  { label: "검은색 서큐버스 헤어" },
  { label: "빨간색 서큐버스 헤어" },
  { label: "주황색 서큐버스 헤어" },
  { label: "검은색 알록달록 헤어" },
  { label: "빨간색 알록달록 헤어" },
  { label: "노란색 알록달록 헤어" },
  { label: "검은색 생머리 그녀 헤어" },
  { label: "노란색 생머리 그녀 헤어" },
  { label: "갈색 생머리 그녀 헤어" },
  { label: "검은색 플뢰르 헤어" },
  { label: "검은색 소프라노 헤어" },
  { label: "검은색 파파야 헤어" },
  { label: "검은색 차밍단발 헤어" },
  { label: "노란색 차밍단발 헤어" },
];

const face = [
  { label: "은한 얼굴" },
  { label: "메이크업 킹 성형" },
  { label: "남자얼굴5 성형" },
  { label: "꿈나라 얼굴" },
  { label: "무심 얼굴" },
  { label: "울림 얼굴" },
  { label: "도발적인 아잉 얼굴" },
  { label: "구르미 얼굴" },
];

const overall = [
  { label: "귀신 옷" },
  { label: "주자로" },
  { label: "청자로" },
  { label: "적자로" },
  { label: "블루 리넥스" },
  { label: "베이지 리넥스" },
  { label: "그린 리넥스" },
  { label: "파란색 설날한복" },
  { label: "초록색 설날한복" },
  { label: "블루 레퀴엠" },
  { label: "레드 레퀴엠" },
  { label: "어둠의 그림자복" },
  { label: "그린 크리시스" },
  { label: "블루 크리시스" },
  { label: "저승사자 한복" },
  { label: "그린 배틀로드" },
  { label: "메소레인저 레드" },
];

const top = [
  { label: "레인보우 탑" },
  { label: "바캉스 체크남방" },
  { label: "스마트 니트조끼" },
  { label: "레드체크더플코트" },
  { label: "러블리 핑크하트티" },
  { label: "블루체크 후디" },
  { label: "그린 타이 셔츠" },
  { label: "캠핑셔츠" },
];

const bottom = [
  { label: "복고교복 바지" },
  { label: "그린 피라테 바지" },
  { label: "다크 피라테 바지" },
  { label: "하와이안 하의" },
  { label: "그린 쉐이드슈트 바지" },
  { label: "블랙 네오스 바지" },
  { label: "밀리터리 카고 반바지" },
  { label: "블루 레깅스" },
  { label: "아이돌스타 체인팬츠" },
  { label: "페페킹의 다크 레골러 바지" },
  { label: "2010 겨울 유치원 옷 바지" },
  { label: "[MS특제] 관급 트레이닝 바지" },
  { label: "주름 호박바지" },
  { label: "랄랄라 도트 팬츠" },
  { label: "파란 세일러 치마" },
  { label: "롤업진" },
  { label: "빨간 세일러 치마" },
];

const shoes = [
  { label: "노란색 닌자 샌들" },
  { label: "파란색 닌자 샌들" },
  { label: "브론즈 체인슈즈" },
  { label: "파란색 고무 장화" },
  { label: "실버 배틀 그리브" },
  { label: "검정색 고무신" },
  { label: "꼬질꼬질한 고무신" },
  { label: "오렌지 미들 래더" },
];

const weapon = [
  { label: "몽둥이" },
  { label: "쇠파이프" },
  { label: "뚜러" },
  { label: "워해머" },
  { label: "너클메이스" },
  { label: "구명 튜브" },
  { label: "도깨비 방망이" },
  { label: "데몬 베인" },
];

const codiPartToData = {
  모자: hat,
  헤어: hair,
  성형: face,
  한벌옷: overall,
  상의: top,
  하의: bottom,
  신발: shoes,
  무기: weapon,
};

export { hat, hair, face, overall, top, bottom, shoes, weapon, codiPartToData };
