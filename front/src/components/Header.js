import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import dinoEgg from "../assets/icons/dino_egg_white.png";
import { ThemeProvider, createTheme } from "@mui/material/styles";

import { useNavigate } from "react-router-dom";

const headerTheme = createTheme({
  palette: {
    primary: {
      main: "#D6A5B6",
    },
  },
  typography: {
    fontFamily: ["PyeongChangPeace-Light", "sans-serif"].join(","),
  },
});

function Header() {
  const navigate = useNavigate();
  return (
    <ThemeProvider theme={headerTheme}>
      <AppBar position="static" color="primary">
        <Toolbar variant="dense">
          <IconButton
            onClick={() => navigate("/recommend")}
            edge="start"
            color="inherit"
            aria-label="menu"
            sx={{ mr: 2 }}>
            <img alt="" src={dinoEgg} width="25px" height="25px" />
          </IconButton>
          <Typography variant="h6" color="white" component="div">
            메이플스토리 코디 추천
          </Typography>
        </Toolbar>
      </AppBar>
    </ThemeProvider>
  );
}
export default Header;
