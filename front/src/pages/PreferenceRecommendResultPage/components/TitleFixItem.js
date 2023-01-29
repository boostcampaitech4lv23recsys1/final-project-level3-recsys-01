import React from "react";
import ResultTitle from "./ResultTitle";
import FixParts from "./FixParts";

function TitleFixItem({ fixPartList }) {
  return (
    <div>
      <ResultTitle></ResultTitle>
      <FixParts fixPartList={fixPartList}></FixParts>
    </div>
  );
}

export default TitleFixItem;
