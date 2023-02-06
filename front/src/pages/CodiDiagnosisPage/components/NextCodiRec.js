import * as React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";

function NextCodiRec() {
  const navigate = useNavigate();
  return (
    <a
      style={{
        color: "#8A37FF",
        fontFamily: "NanumSquareAceb",
        fontSize: "25px",
        cursor: "pointer",
      }}
      onClick={() =>
        window.location.reload(
          navigate("/recommend/preference", { replace: true }),
        )
      }>
      {"코디 추천 받으러 가기"}
    </a>
  );
}

export default NextCodiRec;
