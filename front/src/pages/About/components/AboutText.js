import * as React from "react";
import "./AboutText.css";
import quests from "../../../assets/images/quests.png";
import { Grid } from "@mui/material";

function AboutText() {
  return (
    <div className="AboutText">
      <div className="AboutText-BG">
        <img src={quests} alt="" width={600} height={250}></img>
        <div className="AboutText-text">
          <h2>어바웃 뭐라고 쓰지</h2>
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
      {/* 
      <br />
      <div className="infoVideo">
        <video autoPlay muted loop height="200px">
          <source src="videos/demodemo.mp4" />
        </video>
      </div> */}
    </div>
  );
}

export default AboutText;
