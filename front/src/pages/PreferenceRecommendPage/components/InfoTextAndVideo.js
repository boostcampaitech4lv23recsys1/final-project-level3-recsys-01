import * as React from "react";
import "./InfoTextAndVideo.css";

function InfoTextAndVideo() {
  return (
    <div className="infoTextAndVideo">
      <div className="infoText">
        <h2>부위 별 코디 추천</h2>
        <b>고정하고 싶은 부위를 입력해주세요.</b>
        <br />
        고정하지 않은 부위를 대상으로 추천됩니다.
      </div>
      <br />
      <div className="infoVideo">
        <video autoPlay muted loop height="200px">
          <source src="videos/demodemo.mp4" />
        </video>
      </div>
    </div>
  );
}

export default InfoTextAndVideo;
