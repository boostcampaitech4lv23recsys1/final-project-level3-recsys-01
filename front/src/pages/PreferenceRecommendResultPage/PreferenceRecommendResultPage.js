import TitleFixItem from "./components/TitleFixItem";
import BestCodiTopThree from "./components/BestCodiTopThree";
import RetryButton from "./components/RetryButton";
import LoadingAnimation from "./components/LoadingAnimation";
import "./PreferenceRecommendResultPage.css";
import * as API from "../../api";
import { useState, useEffect } from "react";

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
  const fixPartListKorEng = [];
  let inputParts = {};

  for (let idx = 0; idx < propsParts.length; idx++) {
    if (propsParts[idx]["label"]) {
      inputParts[codiPartEngName[idx]] = Number(propsParts[idx]["index"]);
      fixPartList.push([
        propsParts[idx],
        codiPartName[idx],
        codiPartEngName[idx],
      ]);
      fixPartListKorEng.push(codiPartEngName[idx]);
    } else {
      inputParts[codiPartEngName[idx]] = -1;
    }
  }

  const [recommendData, setRecommendData] = useState({});
  const [loadingPage, setLoadingPage] = useState(false);

  const postCodiPartData = async () => {
    const res = await API.post("inference/submit/MCN", inputParts);
    const data = res.data;
    setRecommendData(data);
    setLoadingPage(true);
  };
  useEffect(() => {
    postCodiPartData();
  }, []);

  if (!loadingPage) {
    return (
      <div className="PRRP">
        <TitleFixItem
          fixPartList={fixPartList}
          loading={loadingPage}></TitleFixItem>
        <center>
          <LoadingAnimation></LoadingAnimation>
        </center>
      </div>
    );
  } else {
    return (
      <div className="PRRP">
        <TitleFixItem
          fixPartList={fixPartList}
          loading={loadingPage}></TitleFixItem>
        <BestCodiTopThree
          fixPartList={fixPartListKorEng}
          recommendData={recommendData}></BestCodiTopThree>
        <RetryButton></RetryButton>
      </div>
    );
  }
}
export default PreferenceRecommendResultPage;
