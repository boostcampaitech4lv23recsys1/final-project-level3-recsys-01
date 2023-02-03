import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import * as API from "../../../api";

function AllCodiPartButton({ codiPart }) {
  const postCodiPartData = async (active) => {
    try {
      const res = await API.get(`inference/submit/newMF/`);
      const data = res.data.items;
      if (active) {
      }
    } catch (err) {
      console.error(err);
    }
  };

  const itemImg = null;
  console.log(codiPart[0]["label"]);
  console.log("dkdkdkdkdkdkdkdkdkdkdkdkdkdkd");

  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>{codiPart}</Typography>
      <Fab aria-label="NotClickable">
        <img alt="" src="" />
      </Fab>
      <Typography>{}</Typography>{" "}
    </Stack>
  );
}
export default AllCodiPartButton;
