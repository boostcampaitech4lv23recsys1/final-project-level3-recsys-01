import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";

function CodiPartButton(props) {
  const isClickable = props.clickable;
  function codiPartButtonMaker(isClickable) {
    if (isClickable) {
      return BasicPopover();
    } else {
      return <Fab aria-label="NotClickable"> Do not cilck! </Fab>;
    }
  }
  const codiPartButton = codiPartButtonMaker(isClickable);
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      <Typography>
        <b> {props.codiPart}</b>
      </Typography>
      {codiPartButton}
    </Stack>
  );
}

export default CodiPartButton;
