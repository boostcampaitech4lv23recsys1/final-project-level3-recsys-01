import React from "react";

function ResultTitle({ loading }) {
  const mainTitle = "AI가 선택한 코디 결과를 보여드릴게요!";
  const loadingTitle = "AI가 옷을 고르고 있어요. 잠시만 기다려주세요!";
  const mainTitleDes =
    "고정한 아이템과 어울리는 아이템 조합을 3가지 보여드려요.\n추천 결과에 대한 피드백이 쌓이면 더욱 똑똑해질 거예요!";
  const loadingDes =
    "내 코디 점수는 몇 점일까? <코디 진단>! \n이렇게 바뀌면 어떨까요? <리뷰 남기기>! \n이 서비스는 누가 만들었을까? <About>!\n";
  return (
    <div className="text-defaultsetting">
      {loading ? (
        <div>
          <h1 style={{ fontFamily: "NanumSquareAceb" }}>{mainTitle}</h1>
          <h3>{mainTitleDes}</h3>
        </div>
      ) : (
        <div>
          <h1 style={{ fontFamily: "NanumSquareAceb" }}>{loadingTitle}</h1>
          <h3>{loadingDes}</h3>
        </div>
      )}
    </div>
  );
}
export default ResultTitle;
