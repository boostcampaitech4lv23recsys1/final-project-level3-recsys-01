import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

function AllCodiPartButton({ partName, codiPart }) {
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <Fab aria-label="NotClickable">
        <img alt="" src={codiPart["gcs_image_url"]} />
      </Fab>
      <Typography>{codiPart["name"]}</Typography>
      <Typography>{partName}</Typography>
    </Stack>
  );
}
export default AllCodiPartButton;
