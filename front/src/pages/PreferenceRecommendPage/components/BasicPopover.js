import * as React from "react";
import { useState, useEffect } from "react";
import Popover from "@mui/material/Popover";
import Fab from "@mui/material/Fab";
import maple_dino from "../../../assets/icons/maple_dino.png";
import BasicSearch from "./BasicSearch";
import BasicButton from "./BasicButton";

function BasicPopover({
  codiPart,
  codiPartData,
  onInputValueChange,
  inputValue,
  inputImage,
  inputId,
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [imagSrc, setImageSrc] = useState(maple_dino);

  // const handleClick = (event) => {
  //   console.log(event);
  //   setAnchorEl(event.target);
  // };

  const handleClick = (newImageSrc) => {
    setImageSrc(newImageSrc);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  // useEffect(() => {
  //   // Re-render BasicButton when inputImage changes
  // }, [inputValue]);

  const open = Boolean(anchorEl);
  const id = open ? "simple-popover" : undefined;

  return (
    <div>
      {/* <Fab aria-label={codiPart}>
        {inputImage === "" ? (
          <img src={maple_dino} alt="" onClick={handleClick} />
        ) : (
          <img src={inputImage} alt="" onClick={handleClick} />
        )}
      </Fab> */}
      <Fab>
        <img src={imagSrc} alt="" onClick={handleClick(inputImage)} />
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
          />
        </div>
      </Popover>
    </div>
  );
}

export default BasicPopover;
