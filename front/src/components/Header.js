import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Typography from "@mui/material/Typography";
import "./Header.css";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import TemporaryDrawer from "./TemporaryDrawer";
import Grid from "@mui/material/Grid";

const headerTheme = createTheme({
  palette: {
    primary: {
      main: "#000000",
    },
  },
});

function Header() {
  const navigate = useNavigate();
  return (
    <ThemeProvider theme={headerTheme}>
      <AppBar
        className="headerAppBar"
        position="static"
        color="primary"
        style={{ height: 70 }}>
        <Grid container>
          <Grid item xs></Grid>
          <Grid item xs={4} className="grid-center">
            <button className="button-title">
              <a
                href="/"
                onClick={(event) => {
                  event.preventDefault();
                  window.location.reload(navigate("/recommend"));
                }}>
                <Typography
                  align="center"
                  variant="h4"
                  color="white"
                  component="div"
                  fontFamily={"PyeongChangPeaceB"}>
                  MESINSA
                </Typography>
              </a>
            </button>
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
