import * as React from "react";
import characterInfo from "../../../assets/images/characterInfo.png";
import SwipeableMemberStepper from "./MemberStepper";

function MemberBox() {
  return (
    <div className="memberBox">
      <div className="memberBox-title"> Members </div>
      <SwipeableMemberStepper className="memberBox-members" />
    </div>
  );
}

export default MemberBox;
