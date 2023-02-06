import React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";
import { textAlign } from "@mui/system";
import { useEffect } from "react";

function RetryButton() {
  const navigate = useNavigate();
  const buttonDes = "다시 추천 받기";
  return (
    <Fab
      variant="extended"
      sx={{
        width: 250,
        backgroundColor: "#8A37FF",
        color: "white",
        fontFamily: "NanumSquareAcb",
        fontSize: 20,
      }}
      onClick={() => {
        window.location.reload(navigate("/", { replace: true }));
      }}
      className="button-retry">
      <a style={{ color: "white" }}>{buttonDes}</a>
    </Fab>
  );
}

export default RetryButton;
