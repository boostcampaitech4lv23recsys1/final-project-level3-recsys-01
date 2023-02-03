import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import * as API from "../../../api";

function FixCodiPartButton({ codiPart }) {
  const itemImg = null;
  console.log(codiPart[0]["label"]);
  console.log("dkdkdkdkdkdkdkdkdkdkdkdkdkdkd");
  const isFix = (codi) => {
    let codiPrint = null;
    if (typeof codi == "string") {
      codiPrint = (
        <div>
          <Typography>{codiPart}</Typography>
          <Fab aria-label="NotClickable">
            <img alt="" src="" />
          </Fab>
          <Typography>{""}</Typography>{" "}
        </div>
      );
    } else {
      codiPrint = (
        <div>
          <Typography>{codiPart[1]}</Typography>
          <Fab aria-label="NotClickable">
            <img alt="" src={codiPart[0]["img"]} />
          </Fab>
          <Typography>{codiPart[0]["label"]}</Typography>
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
