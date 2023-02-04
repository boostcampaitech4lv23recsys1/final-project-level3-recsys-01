import * as React from "react";
import "./CodiPartButton.css";
import Stack from "@mui/material/Stack";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";

function CodiPartButton({ codiPart, inputValue, setInputValue, openPopover }) {
  function handleInputValueChange(newInputValue) {
    let updatedInputValue = {
      label: newInputValue["label"],
      img: newInputValue["img"],
      id: newInputValue["id"],
      category: newInputValue["category"],
      index: newInputValue["index"],
    };
    setInputValue(updatedInputValue);
  }

  return (
    <div className="codiPartButton">
      <BasicPopover
        codiPart={codiPart}
        onInputValueChange={handleInputValueChange}
        inputLabel={inputValue["label"]}
        inputImage={inputValue["img"]}
        inputId={inputValue["id"]}
        inputCategory={inputValue["category"]}
        inputIndex={inputValue["index"]}
        openPopover={openPopover}
      />
    </div>
  );
}

export default CodiPartButton;
