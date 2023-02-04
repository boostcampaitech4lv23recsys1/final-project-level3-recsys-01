import * as React from "react";
import "./InfoTextAndVideo.css";
import quests from "../../../assets/images/quests.png";
import { Grid } from "@mui/material";

function InfoTextAndVideo() {
  return (
    <div className="infoTextAndVideo">
      <div className="infoTextAndVideo-BG">
        <img src={quests} alt="" width={600} height={250}></img>
        <div className="infoTextAndVideo-text">
          <h2>부위 별 코디 추천</h2>
          <b>
            {
              "착용하고 싶은 아이템을 선택해주세요. 선택한 아이템은 고정되며 해당 아이템과 최적의 조합인 다른 부위 아이템이 추천됩니다.\n"
            }
          </b>
          {
            "\n * 상의에서 한벌옷을 선택하면 하의를 선택할 수 없습니다.\n * 모자/신발/무기를 착용하고 싶지 않다면 투명모자/투명신발/투명무기를 선택해주세요."
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

export default InfoTextAndVideo;
