import * as React from "react";
import clickIcon from "../../../assets/icons/click.png";
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
      <Stack direction="row" spacing={8} alignItems="center">
        <CodiPartButton
          codiPart="모자"
          inputValue={inputHat}
          setInputValue={setInputHat}
          openPopover={true}
        />
        <CodiPartButton
          codiPart="헤어"
          inputValue={inputHair}
          setInputValue={setInputHair}
          openPopover={true}
        />
        <CodiPartButton
          codiPart="성형"
          inputValue={inputFace}
          setInputValue={setInputFace}
          openPopover={true}
        />
      </Stack>
      <Stack direction="row" spacing={8} alignItems="center">
        <CodiPartButton
          codiPart="상의"
          inputValue={inputTop}
          setInputValue={setInputTop}
          openPopover={true}
        />
        {inputTop["category"] === "Overall" ? (
          <CodiPartButton
            codiPart="하의"
            inputValue={{ label: "", img: clickIcon, category: "", id: "" }}
            setInputValue={setInputBottom}
            openPopover={false}
          />
        ) : (
          <CodiPartButton
            codiPart="하의"
            inputValue={inputBottom}
            setInputValue={setInputBottom}
            openPopover={true}
          />
        )}
        <CodiPartButton
          codiPart="신발"
          inputValue={inputShoes}
          setInputValue={setInputShoes}
          openPopover={true}
        />
        <CodiPartButton
          codiPart="무기"
          inputValue={inputWeapon}
          setInputValue={setInputWeapon}
          openPopover={true}
        />
      </Stack>
    </Stack>
  );
}

export default CodiPartInputs;
