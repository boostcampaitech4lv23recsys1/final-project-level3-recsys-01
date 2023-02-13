import React from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";

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
      // onClick={() => { // PRP로 돌아가되, 모든 선택 상태 초기화
      //   window.location.reload(
      //     navigate("/recommend/preference", { replace: true }),
      //   );
      // }}
      onClick={() => {
        // PRP로 돌아가되, 선택 상태 유지
        window.history.go(-1);
      }}
      className="button-retry">
      <a
        href="/"
        onClick={(event) => event.preventDefault()}
        style={{ color: "white" }}>
        {buttonDes}
      </a>
    </Fab>
  );
}

export default RetryButton;
