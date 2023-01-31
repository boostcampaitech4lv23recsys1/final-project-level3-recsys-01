import * as React from "react";
import { useState } from "react";

import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";
import { ItemGetFromDB } from "../pages/PreferenceRecommendPage/components/ItemLabel";

function CodiPartButton({ codiPart, inputValue, setInputValue }) {
  function handleInputValueChange(newInputValue) {
    setInputValue(newInputValue);
  }

  const codiPartData = ItemGetFromDB();

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
