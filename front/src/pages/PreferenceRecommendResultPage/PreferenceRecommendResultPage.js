import TitleFixItem from "./components/TitleFixItem";
import BestCodiTopThree from "./components/BsetCodiTopThree";
import RetryButton from "./components/RetryButton";
import "./PreferenceRecommendResultPage.css";

function PreferenceRecommendResultPage(props) {
  const fixPartList = [];
  for (let idx = 0; idx < props.length; idx++) {
    if (props[idx]) {
      fixPartList.push(props[idx]);
    } else {
      continue;
    }
  }
  // const fixPartList = ["헤어", "상의", "하의", "신발", "무기"];
  return (
    <div className="PRRP">
      <TitleFixItem fixPartList={fixPartList}></TitleFixItem>
      <BestCodiTopThree fixPartList={fixPartList}></BestCodiTopThree>
      <RetryButton></RetryButton>
    </div>
  );
}
export default PreferenceRecommendResultPage;
