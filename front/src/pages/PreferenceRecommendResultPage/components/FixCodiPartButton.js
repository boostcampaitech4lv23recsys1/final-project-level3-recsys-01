import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

function FixCodiPartButton({ codiPart }) {
  const isFix = (codi) => {
    let codiPrint = null;
    if (typeof codi == "string") {
      codiPrint = (
        <div>
          <Typography>{codiPart}</Typography>
          <Fab aria-label="NotClickable">
            <img alt="" src="" />
          </Fab>
          <Typography>{""}</Typography> <Typography>{codiPart}</Typography>
        </div>
      );
    } else {
      codiPrint = (
        <div>
          <Fab aria-label="NotClickable">
            <img alt="" src={codiPart[0]["img"]} />
          </Fab>
          <Typography>{codiPart[0]["label"]}</Typography>
          <Typography>{codiPart[1]}</Typography>
        </div>
      );
    }
    return codiPrint;
  };
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      {isFix(codiPart)}
    </Stack>
  );
}
export default FixCodiPartButton;
