import React from "react";
import AllParts from "./AllParts";
import Feedback from "./Feedback";
import { useState } from "react";

function BestCodi({ order, fixPartList, recommendData }) {
  const [feedback, setfeedback] = useState(-1); // default -1 no feedback

  return (
    <div className="block-bestorder">
      <h2 className="button-feedback">Best {order}</h2>
      <Feedback mode={feedback} setfeedback={setfeedback}></Feedback>
      <AllParts
        fixPartList={fixPartList}
        recommendData={recommendData}></AllParts>
    </div>
  );
}

export default BestCodi;
