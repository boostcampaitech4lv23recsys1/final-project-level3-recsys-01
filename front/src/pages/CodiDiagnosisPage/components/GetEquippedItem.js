import * as React from "react";

function GetEquippedItem({
  inputHat,
  inputHair,
  inputFace,
  inputTop,
  inputBottom,
  inputShoes,
  inputWeapon,
}) {
  const partList = [
    inputHat,
    inputHair,
    inputFace,
    inputTop,
    inputBottom,
    inputShoes,
    inputWeapon,
  ];
  const diagnosisInputObject = {};
  for (let part of partList) {
    if (part["category"] == "Overall") {
      diagnosisInputObject["Top"] = part["index"];
    } else {
      diagnosisInputObject[part["category"]] = part["index"];
    }
  }
  return diagnosisInputObject;
}

export default GetEquippedItem;
