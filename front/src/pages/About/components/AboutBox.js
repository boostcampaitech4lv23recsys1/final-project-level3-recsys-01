import * as React from "react";
import quests from "../../../assets/images/quests.png";

function AboutBox() {
  return (
    <div className="aboutBox">
      <div className="aboutBox-title"> About </div>
      <div className="aboutBox-wrap">
        <img className="aboutBox-bg" src={quests} alt="" />
        <div className="aboutBox-text">
          {" "}
          이거 텍스트 정렬 맞추고 있을 바에 걍 이미지로 넣는게 정신 건강에
          좋을듯{" "}
        </div>
      </div>
    </div>
  );
}

export default AboutBox;
