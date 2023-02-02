import * as React from "react";
import { useState, useEffect } from "react";
import Popover from "@mui/material/Popover";
import Fab from "@mui/material/Fab";
import maple_dino from "../../../assets/icons/maple_dino.png";
import BasicSearch from "./BasicSearch";
import Typography from "@mui/material/Typography";

function TopPopover({
  codiPart,
  codiPartData,
  onInputValueChange,
  inputValue,
  inputImage,
  inputId,
  inputCategory,
  setTopInput,
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  const handleClick = (event, inputCategory) => {
    if (event) {
      setAnchorEl(event.target);
      setTopInput(inputCategory);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleInput = (inputCategory) => {
    setTopInput(inputCategory);
  };

  useEffect(() => {
    // Re-render BasicButton when inputImage changes
  }, [inputValue]);

  const open = Boolean(anchorEl);
  const id = open ? "simple-popover" : undefined;

  return (
    <div>
      <Fab aria-label={codiPart}>
        {inputImage === "" ? (
          <img src={maple_dino} alt="" onClick={handleClick(inputCategory)} />
        ) : (
          <img src={inputImage} alt="" onClick={handleClick(inputCategory)} />
        )}
      </Fab>

      <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "center",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "left",
        }}>
        <div
          style={{
            padding: "10px",
            width: "300px",
            height: "55px",
          }}>
          <BasicSearch
            codiPart={codiPart}
            codiPartData={codiPartData}
            onSearchChange={onInputValueChange}
            inputValue={inputValue}
            inputImage={inputImage}
            inputId={inputId}
            inputCategory={inputCategory}
          />
        </div>
      </Popover>
    </div>
  );
}

export default BasicPopover;
