import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import * as API from "../../../api";

function AllCodiPartButton({ partName, codiPart }) {
  const itemImg = null;
  console.log(codiPart);
  console.log("dkdkdkdkdkdkdkdkdkdkdkdkdkdkd");

  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>{codiPart["name"]}</Typography>
      <Fab aria-label="NotClickable">
        <img alt="" src={codiPart["gcs_image_url"]} />
      </Fab>
      <Typography>{partName}</Typography>{" "}
    </Stack>
  );
}
export default AllCodiPartButton;
