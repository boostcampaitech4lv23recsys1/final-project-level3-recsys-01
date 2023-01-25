import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BasicPopover from "../pages/PreferenceRecommendPage/components/BasicPopover";
import BasicSearch from "../pages/PreferenceRecommendPage/components/BasicSearch";

const FloatingBuwiButton = (props) => {
  const isClickable = props.clickable;
  const isSearchable = props.searchable;
  function buwiButtonMaker(isClickable, isSearchable) {
    if (isClickable) {
      return BasicPopover((isSearchable = isSearchable));
    } else {
      return <Fab aria-label="NotClickable"> Do not cilck! </Fab>;
    }
  }
  const buwiButton = buwiButtonMaker(isClickable, isSearchable);
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
