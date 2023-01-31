import * as React from "react";
import { useState } from "react";

import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";
import maple_dino from "../assets/icons/maple_dino.png";
function CodiPartButton({ codiPart, codiPartData }) {
  const [inputValue, setInputValue] = useState("");
  const [inputImage, setInputImage] = useState(maple_dino);
  const [inputId, setInputId] = useState("");

  function handleInputValueChange(newInputValue, newInputImage, newInputId) {
    setInputValue(newInputValue);
    setInputImage(newInputImage);
    setInputId(newInputId);
  }
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>
        <b> {codiPart}</b>
      </Typography>
      <BasicPopover
        codiPart={codiPart}
        codiPartData={codiPartData}
        onInputValueChange={handleInputValueChange}
        inputValue={inputValue}
        inputImage={inputImage}
        inputId={inputId}
      />
      <Typography>{inputValue}</Typography>
    </Stack>
  );
}

export default CodiPartButton;
