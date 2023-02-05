import * as React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";

function NextCodiRec() {
  const navigate = useNavigate();
  return (
    <Fab
      variant="extended"
      onClick={() => window.location.reload(navigate("preference"))}
      sx={{
        marginTop: 5,
        borderRadius: 3,
        border: 1,
        width: 500,
        height: 60,
        backgroundColor: "#8A37FF",
        color: "white",
        fontFamily: "NanumSquareAcb",
        fontSize: 30,
      }}>
      <a style={{ color: "white" }}>{"코디 추천 받으러 가기"}</a>
    </Fab>
  );
}

export default NextCodiRec;
