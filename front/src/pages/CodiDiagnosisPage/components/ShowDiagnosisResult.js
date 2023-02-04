import * as React from "react";

function ShowDiagnosisResult({ diagnosisScore }) {
  const resultScore = Math.ceil(diagnosisScore * 100);
  console.log(diagnosisScore * 100);
  return <p> 당신의 점수는 {resultScore} 점입니다 ㅋㅋ</p>;
}

export default ShowDiagnosisResult;
