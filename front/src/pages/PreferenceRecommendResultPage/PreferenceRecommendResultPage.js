import TitleFixItem from "./components/TitleFixItem";
import BestCodiTopThree from "./components/BestCodiTopThree";
import RetryButton from "./components/RetryButton";
import "./PreferenceRecommendResultPage.css";

function PreferenceRecommendResultPage({
  inputHat,
  inputHair,
  inputFace,
  inputTop,
  inputBottom,
  inputShoes,
  inputWeapon,
}) {
  const propsParts = [
    inputHat,
    inputHair,
    inputFace,
    inputTop,
    inputBottom,
    inputShoes,
    inputWeapon,
  ];
  // for part in parts:
  //   part['label']!="": //사용자가 고정했다는 뜻
  const codiPartName = ["모자", "헤어", "성형", "상의", "하의", "신발", "무기"];
  const codiPartEngName = [
    "Hat",
    "Hair",
    "Face",
    "Top",
    "Bottom",
    "Shoes",
    "Weapon",
  ];
  const fixPartList = [];
  let inputParts = {};

  for (let idx = 0; idx < propsParts.length; idx++) {
    if (propsParts[idx]["label"]) {
      inputParts[codiPartEngName[idx]] = propsParts[idx]["index"];
    } else {
      inputParts[codiPartEngName[idx]] = -1;
    }
  }
  console.log("asdfadsfdsafsfsadfdsafdasfdsafaf");
  console.log(inputParts);

  // const postCodiPartData = async () => {
  //   try {
  //     const res = await API.post(`inference/submit/newMF/${inputParts}`);
  //     const data = res.data.items;
  //   } catch (err) {
  //     console.error(err);
  //   }
  // };
  // for (let idx = 0; idx < 7; idx++) {
  //   if (propsParts[idx]["label"]) {
  //     fixPartList.push([propsParts[idx], codiPartName[idx]]);
  //   } else {
  //     continue;
  //   }
  // }
  console.log("ffffffffffffffffffffffffffffffffffffffffffff");
  console.log(fixPartList);
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
