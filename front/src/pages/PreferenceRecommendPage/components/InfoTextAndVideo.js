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
          <body>
            <b>
              고정하고 싶은 부위를 입력해주세요. 입력되지 않은 부위를 대상으로
              AI의 분석 결과에 따른 추천이 제공됩니다.
            </b>{" "}
            예를 들어 이렇게 저렇게 하시면 됩니다. 모자/신발/무기를 착용하고
            싶지 않다면 투명모자/ 투명신발/ 투명무기를 선택해주세요.
          </body>
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
