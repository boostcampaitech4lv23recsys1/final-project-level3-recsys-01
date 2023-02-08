import * as React from "react";
import Fab from "@mui/material/Fab";
import * as API from "../../../api";

function getEquippedItem({
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
  let diagnosisInputObject = {};
  for (let i = 0; i < partList.length; i++) {
    if (i === 0) {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Hat"] = -1;
      } else {
        diagnosisInputObject["Hat"] = Number(partList[i]["index"]);
      }
    } else if (i === 1) {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Hair"] = -1;
      } else {
        diagnosisInputObject["Hair"] = Number(partList[i]["index"]);
      }
    } else if (i === 2) {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Face"] = -1;
      } else {
        diagnosisInputObject["Face"] = Number(partList[i]["index"]);
      }
    } else if (i === 3) {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Top"] = -1;
      } else {
        diagnosisInputObject["Top"] = Number(partList[i]["index"]);
      }
    } else if (i === 4) {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Bottom"] = -1;
      } else {
        diagnosisInputObject["Bottom"] = Number(partList[i]["index"]);
      }
    } else if (i === 5) {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Shoes"] = -1;
      } else {
        diagnosisInputObject["Shoes"] = Number(partList[i]["index"]);
      }
    } else {
      if (partList[i]["index"] === "") {
        diagnosisInputObject["Weapon"] = -1;
      } else {
        diagnosisInputObject["Weapon"] = Number(partList[i]["index"]);
      }
    }
  }
  return diagnosisInputObject;
}

function GetDiagnosisResult({
  inputHat,
  inputHair,
  inputFace,
  inputTop,
  inputBottom,
  inputShoes,
  inputWeapon,
  diagnosisScore,
  setDiagnosisScore,
  partChange,
  setPartChange,
}) {
  const equippedItem = getEquippedItem({
    inputHat,
    inputHair,
    inputFace,
    inputTop,
    inputBottom,
    inputShoes,
    inputWeapon,
  });
  const sendCodiData = async () => {
    setPartChange(true);
    const res = await API.post("diagnosis/submit/MCN", equippedItem);
    setDiagnosisScore(res.data);
    return diagnosisScore;
  };

  return (
    <div className="getDiagnosisResult">
      <Fab
        variant="extended"
        sx={{
          marginTop: 5,
          borderRadius: 3,
          border: 1,
          width: 500,
          height: 60,
          backgroundColor: "#8A37FF",
          color: "white",
          fontFamily: "NanumSquareAcb",
          fontSize: 30,
        }}
        onClick={() => {
          sendCodiData(); //여기서 셋 파트 체인지를 treu로 바꿔버리면 아무 소용없이 계속 눌리네
        }}
        disabled={partChange}>
        <a
          href="/"
          onClick={(event) => event.preventDefault()}
          style={{ color: "white" }}>
          {"코디 점수 받기"}
        </a>
      </Fab>
    </div>
  );
}

export default GetDiagnosisResult;
