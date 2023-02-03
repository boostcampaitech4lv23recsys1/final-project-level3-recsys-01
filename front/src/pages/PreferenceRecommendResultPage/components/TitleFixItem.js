import React from "react";
import ResultTitle from "./ResultTitle";
import FixParts from "./FixParts";
import resultBackground from "../../../assets/images/resultBackground.png";

function TitleFixItem({ fixPartList }) {
  return (
    <div>
      <img className="titleFixItem-BG" src={resultBackground} alt="" />
      <ResultTitle></ResultTitle>
      <FixParts fixPartList={fixPartList}></FixParts>
    </div>
  );
}

export default TitleFixItem;
