import "./PreferenceRecommendPage.css";
import CodiPartButton from "../../components/CodiPartButton";
import Stack from "@mui/material/Stack";

function PreferenceRecommendPage() {
  return (
    <>
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
      <br />
      <div className="codiPartInput">
        <Stack direction="column" spacing={8} alignItems="center">
          <Stack direction="row" spacing={6} alignItems="center">
            <CodiPartButton codiPart="헤어" clickable={true} />
            <CodiPartButton codiPart="머리" clickable={true} />
            <CodiPartButton codiPart="성형" clickable={true} />
          </Stack>
          <Stack direction="row" spacing={8} alignItems="center">
            <CodiPartButton codiPart="상의" clickable={true} />
            <CodiPartButton codiPart="하의" clickable={true} />
            <CodiPartButton codiPart="신발" clickable={true} />
            <CodiPartButton codiPart="무기" clickable={false} />
          </Stack>
        </Stack>
      </div>
    </>
  );
}
export default PreferenceRecommendPage;
