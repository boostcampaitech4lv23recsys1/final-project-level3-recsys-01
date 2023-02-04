import * as React from "react";
import quests from "../../../assets/images/quests.png";

function AboutBox() {
  return (
    <div className="aboutBox">
      <div className="aboutBox-title"> About </div>
      <div className="aboutBox-wrap">
        <img className="aboutBox-bg" src={quests} alt="" />
        <p className="aboutBox-text"> 안녕 </p>
      </div>
    </div>
  );
}

export default AboutBox;
