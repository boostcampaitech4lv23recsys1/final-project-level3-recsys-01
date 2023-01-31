import React from "react";
import AllParts from "./AllParts";
import good from "../../../assets/images/happy.png";
import bad from "../../../assets/images/calm.png";

function BestCodi({ order, fixPartList }) {
  // add button type later
  return (
    <div className="block-bestorder">
      <h2 className="button-feedback">Best {order}</h2>
      <button className="button-goodbad">
        <img alt="" src={good} width="20" height="20"></img>
      </button>
      <button className="button-goodbad">
        <img alt="" src={bad} width="20" height="20"></img>
      </button>
      <AllParts fixPartList={fixPartList}></AllParts>
    </div>
  );
}

export default BestCodi;
