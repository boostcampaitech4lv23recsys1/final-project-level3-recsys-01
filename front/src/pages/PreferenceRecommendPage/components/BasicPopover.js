import * as React from "react";
import { useState } from "react";
import Popover from "@mui/material/Popover";
import Fab from "@mui/material/Fab";
import BasicSearch from "./BasicSearch";

function BasicPopover({
  codiPart,
  codiPartData,
  onInputValueChange,
  inputLabel,
  inputImage,
  inputId,
  inputCategory,
  openPopover,
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  let open = false;
  if (openPopover) {
    open = Boolean(anchorEl);
  } else {
    open = false;
  }
  const id = open ? "simple-popover" : undefined;

  const handleClick = (event) => {
    setAnchorEl(event.target);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <div>
      <Fab>
        <img src={inputImage} alt="" onClick={handleClick} />
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
            inputValue={inputLabel}
            inputImage={inputImage}
            inputId={inputId}
            inputCategory={inputCategory}
            setAnchorEl={setAnchorEl}
          />
        </div>
      </Popover>
    </div>
  );
}

export default BasicPopover;
