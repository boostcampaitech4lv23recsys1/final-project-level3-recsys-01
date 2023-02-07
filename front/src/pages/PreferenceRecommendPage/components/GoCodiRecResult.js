import * as React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";

function GoCodiRecResult({
  inputHat,
  inputHair,
  inputFace,
  inputTop,
  inputBottom,
  inputShoes,
  inputWeapon,
  partChange,
}) {
  const navigate = useNavigate();
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
    console.log(part);
    if (part["label"] !== "") {
      partChange = false;
    }
  }

  return (
    <Fab
      variant="extended"
      onClick={() => {
        navigate("result");
      }}
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
      disabled={partChange}>
      <a style={{ color: "white" }}>{"코디 추천 받으러 가기"}</a>
    </Fab>
  );
}

export default GoCodiRecResult;
