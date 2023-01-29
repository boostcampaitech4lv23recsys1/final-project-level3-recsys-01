import React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";
import { textAlign } from "@mui/system";

function RetryButton() {
  const navigate = useNavigate();
  const buttonDes = "다시 추천 받기";
  return (
    <Fab
      variant="extended"
      sx={{
        width: 250,
        backgroundColor: "#E5B8C8",
        color: "white",
        fontFamily: "NanumSquareAcb",
        fontSize: 20,
      }}
      className="button-retry">
      <a
        onClick={() => navigate("/", { replace: true })}
        style={{ color: "white" }}>
        {buttonDes}
      </a>
    </Fab>
  );
}

export default RetryButton;
