import React from "react";
import Fab from "@mui/material/Fab";
import Box from "@mui/material/Box";
import dinoEgg from "../../../assets/icons/maple_dino_png.png";
import { useNavigate } from "react-router-dom";

function RecOptionSquare({ optiondes, optionbutton }) {
  const navigate = useNavigate();
  return (
    <Box className="box-recoptionselect">
      <img alt="" src={dinoEgg} width="60px" height="60px" />
      <p className="text-textinbox">{optiondes}</p>
      <Fab
        variant="extended"
        sx={{
          width: 250,
          backgroundColor: "#E5B8C8",
          color: "white",
          fontFamily: "NanumSquareAcb",
          fontSize: 20,
        }}>
        <a onClick={() => navigate("preference")} style={{ color: "white" }}>
          {optionbutton}
        </a>
      </Fab>
    </Box>
  );
}
export default RecOptionSquare;
