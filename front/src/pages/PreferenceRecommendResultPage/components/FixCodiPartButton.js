import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

function FixCodiPartButton({ codiPart, bgImage }) {
  const itemImg = null;
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <img className="fixCodiPartButton-BG" src={bgImage} alt="" />
      <a className="fixCodiPartButton-item" aria-label="NotClickable">
        {itemImg}
      </a>
      <Typography className="fixCodiPartButton-part">
        <b>{codiPart}</b>
      </Typography>
    </Stack>
  );
}
export default FixCodiPartButton;
