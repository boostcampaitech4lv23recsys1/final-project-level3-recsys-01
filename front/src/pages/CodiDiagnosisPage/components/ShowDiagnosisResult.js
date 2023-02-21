import * as React from "react";
import NextCodiRec from "./NextCodiRec";

function ShowDiagnosisResult({ diagnosisScore }) {
  const resultScore = Math.ceil(diagnosisScore * 100);
  let scoreDes = null;
  if (resultScore < 25) {
    scoreDes = "정말 개성있는 코디네요! AI가 더 개성있는 옷을 골라드릴게요!";
  } else if (25 <= resultScore && resultScore < 50) {
    scoreDes = "정말 멋있는 코디네요! AI와 함께 더 멋진 옷을 골라볼까요?";
  } else if (50 <= resultScore && resultScore < 75) {
    scoreDes = "오... 코디 좀 하시는데요? AI와 코디 실력을 겨뤄보세요!";
  } else if (75 <= resultScore && resultScore <= 100) {
    scoreDes = "코디를 정말 잘하시네요! AI에게 한 수 가르쳐 주세요!";
  }
  return (
    <div className="diagnosisResult">
      <h2> 당신의 점수는 {resultScore} 점입니다!</h2> <p>{scoreDes}</p>
      <NextCodiRec></NextCodiRec>
    </div>
  );
}

export default ShowDiagnosisResult;
