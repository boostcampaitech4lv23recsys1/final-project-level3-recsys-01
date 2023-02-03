import * as React from "react";

function CodiSimulator({
  inputHat,
  inputHair,
  inputFace,
  inputTop,
  inputBottom,
  inputShoes,
  inputWeapon,
}) {
  let baseURL =
    "https://maplestory.io/api/character/%7B%22itemId%22%3A2000%2C%22region%22%3A%22KMST%22%2C%22version%22%3A%221149%22%7D%2C%7B%22itemId%22%3A12000%2C%22region%22%3A%22KMST%22%2C%22version%22%3A%221149%22%7D%2C";
  const addURLBefore = "%7B%22itemId%22%3A";
  const addURLAfter =
    "%2C%22region%22%3A%22KMST%22%2C%22version%22%3A%221149%22%7D%2C";
  const endURL =
    "/stand1/animated?showears=false&showLefEars=false&showHighLefEars=undefined&resize=3&name=&flipX=false&bgColor=0,0,0,0";
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
    if (part["id"] !== "") {
      baseURL = baseURL + addURLBefore + part["id"] + addURLAfter;
    }
  }
  const finalURL = baseURL + endURL;
  return <img src={finalURL} alt="" />;
}

export default CodiSimulator;
