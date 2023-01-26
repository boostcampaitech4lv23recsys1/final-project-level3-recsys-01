import React from "react";
import { useState } from "react";
import Popover from "@mui/material/Popover";
import Fab from "@mui/material/Fab";
import BasicSearch from "./BasicSearch";

function BasicPopover() {
  const [anchorEl, setAnchorEl] = useState(null);

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);
  const id = open ? "simple-popover" : undefined;

  return (
    <div>
      <Fab aria-label="Click!" onClick={handleClick}>
        Click!
      </Fab>
      <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "centerehr",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "left",
        }}>
        <div
          style={{
            padding: "10px",
            width: "300px",
            height: "200px",
          }}>
          <BasicSearch />
        </div>
      </Popover>
    </div>
  );
}

export default BasicPopover;
