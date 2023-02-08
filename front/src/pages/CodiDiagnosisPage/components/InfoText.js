import * as React from "react";
import "./InfoText.css";
import quests from "../../../assets/images/quests.png";

function InfoText() {
  return (
    <div className="infoText">
      <div className="infoText-BG">
        <img src={quests} alt="" width={600} height={250}></img>
        <div className="infoText-text">
          <h2>코디 조합 진단하기</h2>
          <b>
            {
              "착용하고 있는 아이템을 선택해주세요. AI가 아이템들의 조합을 평가합니다. 100점에 도전해보세요!\n"
            }
          </b>
          {
            "\n * 상의에서 한벌옷을 선택하면 하의를 선택할 수 없습니다.\n * 모자/신발/무기를 착용하고 있지 않다면 투명모자/투명신발/투명무기를 선택해주세요."
          }
        </div>
      </div>
    </div>
  );
}

export default InfoText;
