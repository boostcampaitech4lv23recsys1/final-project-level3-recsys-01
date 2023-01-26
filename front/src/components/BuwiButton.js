import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";

const FloatingBuwiButton = (props) => {
  const isClickable = props.clickable;
  function buwiButtonMaker(isClickable) {
    if (isClickable) {
      return BasicPopover();
    } else {
      return <Fab aria-label="NotClickable"> Do not cilck! </Fab>;
    }
  }
  const buwiButton = buwiButtonMaker(isClickable);
  return (
    <Stack direction="column" spacing={1}>
      <Typography>
        <b> {props.buwi}</b>
      </Typography>
      {buwiButton}
    </Stack>
  );
};

export default FloatingBuwiButton;
