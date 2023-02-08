import * as React from "react";
import { useNavigate } from "react-router-dom";

function NextCodiRec() {
  const navigate = useNavigate();
  return (
    <a
      href="/"
      style={{
        color: "#8A37FF",
        fontFamily: "NanumSquareAceb",
        fontSize: "25px",
        cursor: "pointer",
      }}
      onClick={(event) => {
        event.preventDefault();
        navigate("/recommend/preference", { replace: true });
      }}>
      {"이 상태에서 추천 받아보기"}
    </a>
  );
}

export default NextCodiRec;
