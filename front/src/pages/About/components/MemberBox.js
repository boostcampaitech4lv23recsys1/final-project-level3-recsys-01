import * as React from "react";
import SwipeableMemberStepper from "./MemberStepper";

function MemberBox() {
  return (
    <div className="memberBox">
      <div className="memberBox-title"> Members </div>
      <SwipeableMemberStepper />
    </div>
  );
}

export default MemberBox;
