import "./PreferenceRecommendPage.css";
import FloatingBuwiButton from "../../components/BuwiButton";
import Stack from "@mui/material/Stack";

function PreferenceRecommendPage() {
  return (
    <>
      <div className="InfoText">
        <h2>부위 별 코디 추천</h2>
        <b>고정하고 싶은 부위를 입력해주세요.</b>
        <br />
        고정하지 않은 부위를 대상으로 추천됩니다.
      </div>
      <br />
      <div className="InfoVideo">
        <video autoPlay muted loop height="200px">
          <source src="videos/demodemo.mp4" />
        </video>
      </div>
      <br />
      <div className="BuwiItemInput">
        <Stack direction="column" spacing={8} alignItems="center">
          <Stack direction="row" spacing={6} alignItems="center">
            <FloatingBuwiButton buwi="헤어" />
            <FloatingBuwiButton buwi="머리" />
            <FloatingBuwiButton buwi="성형" />
          </Stack>
          <Stack direction="row" spacing={8} alignItems="center">
            <FloatingBuwiButton buwi="상의" />
            <FloatingBuwiButton buwi="하의" />
            <FloatingBuwiButton buwi="신발" />
            <FloatingBuwiButton buwi="무기" />
          </Stack>
        </Stack>
      </div>
    </>
  );
}
export default PreferenceRecommendPage;
