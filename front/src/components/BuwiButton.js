import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { usePreviousProps } from "@mui/utils";

const FloatingBuwiButton = (props) => (
  <Stack direction="column" spacing={1}>
    <Typography>
      <b> {props.buwi}</b>
    </Typography>
    <Fab aria-label="Click!">Click!</Fab>
  </Stack>
);

export default FloatingBuwiButton;
