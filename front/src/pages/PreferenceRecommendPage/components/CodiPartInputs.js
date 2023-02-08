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
  setPartChange,
}) {
  const defaultFixObject = {
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  };
  return (
    <div>
      <Stack direction="column" spacing={8} alignItems="center" sx={{ p: 8 }}>
        <Stack direction="row" spacing={8} alignItems="center">
          <CodiPartButton
            codiPart="모자"
            inputValue={inputHat}
            setInputValue={setInputHat}
            openPopover={true}
            setPartChange={setPartChange}
          />
          <CodiPartButton
            codiPart="헤어"
            inputValue={inputHair}
            setInputValue={setInputHair}
            openPopover={true}
            setPartChange={setPartChange}
          />
          <CodiPartButton
            codiPart="성형"
            inputValue={inputFace}
            setInputValue={setInputFace}
            openPopover={true}
            setPartChange={setPartChange}
          />
        </Stack>
        <Stack direction="row" spacing={8} alignItems="center">
          <CodiPartButton
            codiPart="상의"
            inputValue={inputTop}
            setInputValue={setInputTop}
            openPopover={true}
            setPartChange={setPartChange}
          />
          {inputTop["category"] === "Overall" ? (
            <CodiPartButton
              codiPart="하의"
              inputValue={defaultFixObject}
              setInputValue={setInputBottom}
              openPopover={false}
              setPartChange={setPartChange}
            />
          ) : (
            <CodiPartButton
              codiPart="하의"
              inputValue={inputBottom}
              setInputValue={setInputBottom}
              openPopover={true}
              setPartChange={setPartChange}
            />
          )}
          <CodiPartButton
            codiPart="신발"
            inputValue={inputShoes}
            setInputValue={setInputShoes}
            openPopover={true}
            setPartChange={setPartChange}
          />
          <CodiPartButton
            codiPart="무기"
            inputValue={inputWeapon}
            setInputValue={setInputWeapon}
            openPopover={true}
            setPartChange={setPartChange}
          />
          {console.log(inputTop, inputBottom)}
        </Stack>
      </Stack>
      <button
        className="button-reload"
        style={{
          borderRadius: 30,
          width: 150,
          height: 30,
          border: 1,
          backgroundColor: "#b9b9b9",
          color: "white",
          fontFamily: "NanumSquareAcb",
          fontSize: 20,
          cursor: "pointer",
        }}
        onClick={() => {
          setInputHat(defaultFixObject);
          setInputHair(defaultFixObject);
          setInputFace(defaultFixObject);
          setInputTop(defaultFixObject);
          setInputBottom(defaultFixObject);
          setInputShoes(defaultFixObject);
          setInputWeapon(defaultFixObject);
          setPartChange(true);
        }}>
        All Reset
      </button>
    </div>
  );
}

export default CodiPartInputs;
