import * as React from "react";
import { useState } from "react";
import Popover from "@mui/material/Popover";
import Fab from "@mui/material/Fab";
import BasicSearch from "./BasicSearch";
import click from "../../../assets/icons/click.png";
import basicItem from "../../../assets/images/basicItem.png";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import "./BasicPopover.css";

function BasicPopover({
  codiPart,
  codiPartData,
  onInputValueChange,
  inputLabel,
  inputImage,
  inputId,
  inputCategory,
  inputIndex,
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
    <div className="basicPopover">
      {/* <Fab>
        <img src={inputImage} alt="" onClick={handleClick} />
      </Fab> */}

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
            inputIndex={inputIndex}
            setAnchorEl={setAnchorEl}
          />
        </div>
      </Popover>
      <Stack
        className="itemSelectionStack"
        direction="column"
        spacing={0}
        alignItems="center">
        <div className="itemSelection">
          <img
            className="itemSelection-bg"
            src={basicItem}
            onClick={handleClick}
            alt=""
          />
          {inputId === "" ? (
            <p className="itemSelection-text" onClick={handleClick}>
              {" "}
              Click Here!{" "}
            </p>
          ) : (
            <img
              className="itemSelection-img"
              src={inputImage}
              onClick={handleClick}
              alt={inputLabel}
            />
          )}
        </div>
        <Typography className="itemSelectionStack-codiPart">
          <b> {codiPart}</b>
        </Typography>
        <Typography className="itemSelectionStack-itemLabel">
          {inputLabel}
        </Typography>
      </Stack>
    </div>
  );
}

export default BasicPopover;
