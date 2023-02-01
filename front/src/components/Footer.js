import React from "react";
import Box from "@mui/material/Box";
import "./Footer.css";

function StickyFooter() {
  const teamDescriptionOne = "Contact | ";
  const teamMail = "gongryongal1@gmail.com";
  const teamDescriptionTwo =
    "\n네이버 부스트캠프 AI Tech 4기 RecSys Track 공룡알 팀의 최종 프로젝트입니다.";
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
      }}>
      <Box
        component="footer"
        sx={{
          py: 3,
          px: 2,
          mt: "auto",
          backgroundColor: "black",
          marginTop: 10,
        }}>
        <div className="text-footer">
          {teamDescriptionOne}
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="mailto:gongryongal1@gmail.com"
            className="text-hyperlink"
            style={{ color: "white" }}>
            {teamMail}
          </a>
          {teamDescriptionTwo}
        </div>
      </Box>
    </Box>
  );
}
export default StickyFooter;
