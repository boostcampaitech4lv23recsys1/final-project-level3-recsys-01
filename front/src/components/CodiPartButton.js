import * as React from "react";
import { useState, useEffect } from "react";
import "./CodiPartButton.css";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";

function CodiPartButton({
  codiPart,
  inputValue,
  setInputValue,
  openPopover,
  numberState,
  setNumberState,
}) {
  const defaultFixObject = {
    label: "",
    img: null,
    id: "",
    category: "",
    index: "",
  };

  function handleInputValueChange(newInputValue) {
    let updatedInputValue = {
      label: newInputValue["label"],
      img: newInputValue["img"],
      id: newInputValue["id"],
      category: newInputValue["category"],
      index: newInputValue["index"],
    };
    setInputValue(updatedInputValue);
    setNumberState(numberState + 1);
  }
  useEffect(() => {
    if (openPopover === false) {
      setInputValue(defaultFixObject);
      setNumberState(numberState + 1);
    }
  }, [openPopover]); // 상의가 Overall인 경우

  return (
    <div className="codiPartButton">
      <BasicPopover // 여기에서 아이템 선택 + 이름 띄워주기까지 다 함
        codiPart={codiPart}
        onInputValueChange={handleInputValueChange}
        inputLabel={inputValue["label"]}
        inputImage={inputValue["img"]}
        inputId={inputValue["id"]}
        inputCategory={inputValue["category"]}
        inputIndex={inputValue["index"]}
        openPopover={openPopover}
      />
      <button
        style={{
          position: "relative",
          top: 10,
          left: 15,
          marginLeft: 20,
          borderRadius: 30,
          width: 50,
          height: 20,
          border: 1,
          backgroundColor: "#b9b9b9",
          color: "white",
          fontFamily: "NanumSquareAcb",
          fontSize: 15,
          textAlign: "center",
          cursor: "pointer",
        }}
        onClick={() => {
          if (inputValue["category"] === "Overall") {
            setNumberState(numberState - 2); // 여기에서 Object.values(partState)의 sum 값으로 바꿔줘야 함
          } else {
            setNumberState(numberState - 1);
          }
          setInputValue(defaultFixObject);
        }}>
        Reset
      </button>
    </div>
  );
}

export default CodiPartButton;
