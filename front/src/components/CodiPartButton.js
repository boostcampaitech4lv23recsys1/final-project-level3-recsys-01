import * as React from "react";
import { useState } from "react";

import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
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
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>
        <b> {codiPart}</b>
      </Typography>
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
      <Typography>{inputValue["label"]}</Typography>
    </Stack>
  );
}

export default CodiPartButton;
