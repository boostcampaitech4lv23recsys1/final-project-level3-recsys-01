import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import mapleDino from "../assets/icons/maple_dino.png";
import "./Header.css";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";

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
        <Toolbar variant="dense">
          <IconButton
            onClick={() => navigate("/recommend")}
            edge="start"
            color="inherit"
            aria-label="menu">
            {/* <img alt="" src={mapleDino} width="25px" height="25px" /> */}
          </IconButton>
          <Typography align="center" variant="h5" color="white" component="div">
            MESINSA
          </Typography>
        </Toolbar>
      </AppBar>
    </ThemeProvider>
  );
}
export default Header;
