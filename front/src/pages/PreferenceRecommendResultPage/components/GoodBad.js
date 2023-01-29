import React from "react";
import good from "../../../assets/images/happy.png";
import bad from "../../../assets/images/calm.png";

function GoodBad() {
  return (
    <div className="button-twofeedback">
      <button className="button-goodbad">
        <img alt="" src={good} width="20" height="20"></img>
      </button>
      <button className="button-goodbad">
        <img alt="" src={bad} width="20" height="20"></img>
      </button>
    </div>
  );
}

export default GoodBad;
