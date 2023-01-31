import * as React from "react";
import Stack from "@mui/material/Stack";
import CodiPartButton from "../../../components/CodiPartButton";
import "./CodiPartInputs.css";

function CodiPartInputs({
  inputHat,
  setInputHat,
  inputHair,
  setInputHair,
  inputFace,
  setInputFace,
  inputTop,
  setInputTop,
  inputBottom,
  setInputBottom,
  inputShoes,
  setInputShoes,
  inputWeapon,
  setInputWeapon,
}) {
  return (
    <Stack direction="column" spacing={8} alignItems="center" sx={{ p: 8 }}>
      <Stack direction="row" spacing={6} alignItems="center">
        <CodiPartButton
          codiPart="모자"
          inputValue={inputHat}
          setInputValue={setInputHat}
        />
        <CodiPartButton
          codiPart="헤어"
          inputValue={inputHair}
          setInputValue={setInputHair}
        />
        <CodiPartButton
          codiPart="성형"
          inputValue={inputFace}
          setInputValue={setInputFace}
        />
      </Stack>
      <Stack direction="row" spacing={8} alignItems="center">
        <CodiPartButton
          codiPart="상의"
          inputValue={inputTop}
          setInputValue={setInputTop}
        />
        {inputTop["category"] === "Overall" ? (
          <CodiPartButton
            codiPart="하의"
            inputValue={inputBottom}
            setInputValue={setInputBottom}
          />
        ) : (
          <CodiPartButton
            codiPart="하의"
            inputValue={inputBottom}
            setInputValue={setInputBottom}
          />
        )}
        <CodiPartButton
          codiPart="신발"
          inputValue={inputShoes}
          setInputValue={setInputShoes}
        />
        <CodiPartButton
          codiPart="무기"
          inputValue={inputWeapon}
          setInputValue={setInputWeapon}
        />
      </Stack>
    </Stack>
  );
}

export default CodiPartInputs;
