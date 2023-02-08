import * as React from "react";
import quests from "../../../assets/images/quests.png";

function AboutBox() {
  return (
    <div className="aboutBox">
      <div className="aboutBox-title"> About </div>
      <div className="aboutBox-wrap">
        <img className="aboutBox-bg" src={quests} alt="" />
        <div className="aboutBox-text">
          <h2>Team. 공룡알</h2>
          안에 무엇이 있는지 알 수 없는 알과 같은 미지의 상태로 시작했으나
          최종적으로는 모두 무시무시한 공룡이 되고자 ‘공룡알’이 되었습니다. 지식
          공유와 문서화를 강점으로 가진 팀입니다. 프로젝트 진행 과정을 자세하게
          살펴보고 싶으시다면{" "}
          <a href="https://41ow1ives.notion.site/Final-Project-7297be317b9340b5a7c18d70c010f783">
            "팀 노션"
          </a>
          에 방문해주세요. 감사합니다.
        </div>
      </div>
    </div>
  );
}

export default AboutBox;
