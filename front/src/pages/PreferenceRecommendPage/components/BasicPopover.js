import React from "react";
import {useState} from React
import Popover from "@mui/material/Popover";
import Fab from "@mui/material/Fab";

import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";

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

  const top5Songs = [
    { label: "Organise" },
    { label: "Joha" },
    { label: "Terminator" },
    { label: "Dull" },
    { label: "Nzaza" },
  ];
  const top5SongsLength = top5Songs.length;
  const searchBox = (
    <Autocomplete
      disablePortal
      id="combo-box-demo"
      options={top5Songs}
      sx={{ width: 300 }}
      renderInput={(params) => <TextField {...params} label="Songs" />}
    />
  );

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
          horizontal: "center",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "left",
        }}>
        <div
          style={{
            width: "300px",
            height: "200px",
          }}>
          {searchBox}
        </div>
      </Popover>
    </div>
  );
}

export default BasicPopover;
