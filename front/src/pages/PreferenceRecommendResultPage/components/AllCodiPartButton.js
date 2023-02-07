import React from "react";
import Typography from "@mui/material/Typography";

function AllCodiPartButton({ partName, codiPart }) {
  return (
    <div>
      <img alt="" src={codiPart["gcs_image_url"]} width="70%" height="70%" />
      <Typography>{codiPart["name"]}</Typography>
      <Typography>{partName}</Typography>
    </div>
  );
}
export default AllCodiPartButton;
