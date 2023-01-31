import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import "./Header.css";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import TemporaryDrawer from "./TemporaryDrawer";
import Grid from "@mui/material/Grid";

const headerTheme = createTheme({
  palette: {
    primary: {
      main: "#D6A5B6",
    },
  },
  typography: {
    fontFamily: ["PyeongChangPeace-Bold", "sans-serif"].join(","),
  },
});

function Header() {
  const navigate = useNavigate();
  return (
    <ThemeProvider theme={headerTheme}>
      <AppBar className="headerAppBar" position="static" color="primary">
        <Grid container>
          <Grid item xs></Grid>
          <Grid item xs={4} className="grid-center">
            <IconButton
              onClick={() => navigate("/recommend")}
              edge="start"
              color="inherit"
              aria-label="menu">
              <Typography
                align="center"
                variant="h5"
                color="white"
                component="div">
                MESINSA
              </Typography>
            </IconButton>
          </Grid>
          <Grid item xs={4} className="grid-drawer">
            <TemporaryDrawer></TemporaryDrawer>
          </Grid>
        </Grid>
      </AppBar>
    </ThemeProvider>
  );
}
export default Header;
