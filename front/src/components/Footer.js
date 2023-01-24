import React from "react";
import Box from "@mui/material/Box";

export default function StickyFooter() {
  const teamDescription =
    "Contact | gongryongal1@gmail.com \n 네이버 부스트캠프 AI Tech 4기 RecSys Track 공룡알 팀의 최종 프로젝트입니다.";
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
      }}>
      <Box
        component="footer"
        sx={{
          py: 3,
          px: 2,
          mt: "auto",
          backgroundColor: "#FCCEAD",
        }}>
        <div className="text-footer">{teamDescription}</div>
      </Box>
    </Box>
  );
}
