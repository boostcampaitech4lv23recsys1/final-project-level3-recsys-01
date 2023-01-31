import * as React from "react";
import { useState } from "react";

import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";
import { ItemGetFromDB } from "../pages/PreferenceRecommendPage/components/ItemLabel";

function CodiPartButton(props) {
  const [inputValue, setInputValue] = useState("");
  const [inputImage, setInputImage] = useState("");
  const [inputId, setInputId] = useState("");

  const codiPartData = ItemGetFromDB(props.codiPart);

  function handleInputValueChange(newInputValue, newInputImage, newInputId) {
    setInputValue(newInputValue);
    setInputImage(newInputImage);
    setInputId(newInputId);
  }
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>
        <b> {props.codiPart}</b>
      </Typography>
      <BasicPopover
        codiPart={props.codiPart}
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
