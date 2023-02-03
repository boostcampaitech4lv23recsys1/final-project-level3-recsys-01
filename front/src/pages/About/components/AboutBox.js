import * as React from "react";
import quests from "../../../assets/images/quests.png";

function AboutBox() {
  return (
    <div class="aboutBox">
      <div class="aboutBox-title"> About </div>
      <div class="aboutBox-wrap">
        <img class="aboutBox-bg" src={quests} alt="" />
        <p class="aboutBox-text"> 안녕 </p>
      </div>
    </div>
  );
}

export default AboutBox;
