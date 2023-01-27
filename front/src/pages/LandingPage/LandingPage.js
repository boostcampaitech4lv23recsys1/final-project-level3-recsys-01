import "./LandingPage.css";

import TitleDescription from "../../components/TitleDescription";
import RoundSquare from "./components/RoundSquare";
import LinkButton from "./components/LinkButton";
import dinoEgg from "../../assets/icons/dino_egg_white.png";
import github from "../../assets/icons/github.png";

function LandingPage() {
  const titleOne = "Home";
  const titleOneDes =
    "이 기능에 대해 뭐라고\n 한 두 줄 정도로 예쁘고 간결하게 설명하기";
  const preferDes =
    "부위별 코디 추천을 설명 \n 위 아이콘은 부위별 코디추천 아이콘";
  const preferMoveDes = "코디 추천 받기";
  const titleTwo = "Teams";
  const titleTwoDes = "어떤 가치를 어쩌구\n 저쩌구\n 호롤롤루";
  const githubLink = "https://www.naver.com/";
  const linkname = "Team Github";

  return (
    <div>
      <TitleDescription
        title={titleOne}
        description={titleOneDes}></TitleDescription>
      <br></br>
      <center>
        <div className="body-one-margin">
          <RoundSquare
            icon={dinoEgg}
            description={preferDes}
            movedescription={preferMoveDes}></RoundSquare>
        </div>
        <div className="body-one-margin">
          <RoundSquare
            icon={dinoEgg}
            description={preferDes}
            movedescription={preferMoveDes}></RoundSquare>
        </div>
      </center>
      <TitleDescription
        title={titleTwo}
        description={titleTwoDes}></TitleDescription>
      <center>
        <div className="body-two-margin">
          <img src={github} width="50px" height="50px" className="img-margin" />
        </div>
        <LinkButton
          link={githubLink}
          displaylink={linkname}
          className="body-two-margin"></LinkButton>
      </center>
    </div>
  );
}
export default LandingPage;
