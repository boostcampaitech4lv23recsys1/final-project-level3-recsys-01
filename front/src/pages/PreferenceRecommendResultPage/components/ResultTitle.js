import React from "react";

function ResultTitle() {
  const mainTitle = "Result Title";
  const mainTitleDes = "당신이 고정한 아이템과 어울리는 추천 아이템은.....?";
  return (
    <div className="text-defaultsetting">
      <h1 style={{ fontFamily: "NanumSquareAceb" }}>{mainTitle}</h1>
      <h3>{mainTitleDes}</h3>
    </div>
  );
}
export default ResultTitle;
