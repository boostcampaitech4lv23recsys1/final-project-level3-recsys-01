import React from "react";
import ResultTitle from "./ResultTitle";
import FixParts from "./FixParts";
import resultBackground from "../../../assets/images/resultBackground.png";

function TitleFixItem({ fixPartList, loading }) {
  return (
    <div>
      <img className="titleFixItem-BG" src={resultBackground} alt="" />
      <ResultTitle loading={loading}></ResultTitle>
      <FixParts fixPartList={fixPartList}></FixParts>
    </div>
  );
}

export default TitleFixItem;
