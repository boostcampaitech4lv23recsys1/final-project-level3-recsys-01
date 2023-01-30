import * as React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";

function GoCodiDignosis() {
  const navigate = useNavigate();
  return (
    <div>
      <Fab
        variant="extended"
        sx={{
          marginTop: 2,
          borderRadius: 0,
          width: 200,
          backgroundColor: "white",
          color: "black",
          fontFamily: "NanumSquareAcb",
          fontSize: 15,
          margin: "0auto",
          boxShadow: 0,
        }}
        className="button-gorec">
        <a onClick={() => navigate("preference")}>
          {"코디 진단부터 받으러 가기"}
        </a>
      </Fab>
    </div>
  );
}

export default GoCodiDignosis;
