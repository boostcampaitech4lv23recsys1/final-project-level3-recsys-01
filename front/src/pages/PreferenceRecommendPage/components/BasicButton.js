import * as React from "react";
import Fab from "@mui/material/Fab";

function BasicButton({ codiPart, inputImage, handleClick }) {
  let basicButton = (
    <Fab aria-label={codiPart} onClick={handleClick}>
      Click!
    </Fab>
  );
  if (inputImage !== "") {
    basicButton = (
      <Fab aria-label={codiPart} onclick={handleClick}>
        <img src={inputImage} alt="" />
      </Fab>
    );
  }
  return basicButton;
}

export default BasicButton;
