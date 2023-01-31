import * as React from "react";
import { useState } from "react";

import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";
import maple_dino from "../assets/icons/maple_dino.png";

function CodiPartButton({ codiPart, codiPartData, inputValue, setInputValue }) {
  function handleInputValueChange(newInputValue, newInputImage, newInputId) {
    setInputValue(newInputValue);
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
        inputLabel={inputValue["label"]}
        inputImage={inputValue["img"]}
        inputId={inputValue["id"]}
        inputCategory={inputValue["category"]}
      />
      <Typography>{inputValue["label"]}</Typography>
    </Stack>
  );
}

export default CodiPartButton;
