import * as React from "react";
import Stack from "@mui/material/Stack";
import { useState, useEffect } from "react";
import Typography from "@mui/material/Typography";

function TopBottomInput(props) {
  const [topInput, setTopInput] = useState("Top");
  const [inputValue, setInputValue] = useState("");
  const [inputImage, setInputImage] = useState("");
  const [inputId, setInputId] = useState("");
  const [inputCategory, setInputCategory] = useState("");

  function handleInputValueChange(
    newInputValue,
    newInputImage,
    newInputId,
    newInputCategory,
  ) {
    setInputValue(newInputValue);
    setInputImage(newInputImage);
    setInputId(newInputId);
    setInputCategory(newInputCategory);
  }

  const TopButton = () => {
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>
        <b> {"상의"}</b>
      </Typography>
      <TopPopover
        codiPart={"상의"}
        codiPartData={codiPartData}
        onInputValueChange={handleInputValueChange}
        inputValue={inputValue}
        inputImage={inputImage}
        inputId={inputId}
        inputCategory={inputCategory}
        setTopInput={setTopInput}
      />
      <Typography>{inputValue}</Typography>
    </Stack>;
  };

  const BottomButton = (topInput) => {
    return (
      topInput != "Overall"
      ?(<Stack direction="column" spacing={1} alignItems="center">
        <Typography>
          <b> {"하의"}</b>
        </Typography>
        <BottomPopover
          codiPart={"하의"}
          codiPartData={codiPartData}
          onInputValueChange={handleInputValueChange}
          inputValue={inputValue}
          inputImage={inputImage}
          inputId={inputId}
          inputCategory={inputCategory}
        />
        <Typography>{inputValue}</Typography>
      </Stack>)
    :()
    );
  };
}
