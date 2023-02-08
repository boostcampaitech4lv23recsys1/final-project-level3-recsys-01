import * as React from "react";
// import simulatorBg from "../assets/images/simulatorBg.png";

function CodiSimulator({
  inputHat,
  inputHair,
  inputFace,
  inputTop,
  inputBottom,
  inputShoes,
  inputWeapon,
  size,
  isResult,
}) {
  let baseURL =
    "https://maplestory.io/api/character/%7B%22itemId%22%3A2000%2C%22region%22%3A%22KMST%22%2C%22version%22%3A%221149%22%7D%2C%7B%22itemId%22%3A12000%2C%22region%22%3A%22KMST%22%2C%22version%22%3A%221149%22%7D%2C";
  const addURLBefore = "%7B%22itemId%22%3A";
  const addURLAfter =
    "%2C%22region%22%3A%22KMST%22%2C%22version%22%3A%221149%22%7D%2C";
  const endURL =
    "/stand1/animated?showears=false&showLefEars=false&showHighLefEars=undefined&resize=" +
    size +
    "&name=&flipX=false&bgColor=0,0,0,0";
  const partList = [
    inputHat,
    inputHair,
    inputFace,
    inputTop,
    inputBottom,
    inputShoes,
    inputWeapon,
  ];
  for (let part of partList) {
    if (isResult) {
      if (String(part["item_id"]) !== "") {
        baseURL =
          baseURL + addURLBefore + String(part["item_id"]) + addURLAfter;
      }
    } else {
      if (part["id"] !== "") {
        baseURL = baseURL + addURLBefore + part["id"] + addURLAfter;
      }
    }
  }
  const finalURL = baseURL + endURL;
  return (
    <div className="codiSimulator-container">
      {/* <img className="codiSimulator-bg" src={simulatorBg} alt=""></img> */}
      <img src={finalURL} alt="" className="codiSimulator-simulator"></img>
    </div>
  );
}

export default CodiSimulator;
