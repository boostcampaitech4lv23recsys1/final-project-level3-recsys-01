import React from "react";
import Fab from "@mui/material/Fab";

function GithubLinkButton() {
  const teamGithubLink = "https://www.naver.com/";
  const teamGithubLinkDes = "Team Github";
  return (
    <Fab
      variant="extended"
      sx={{
        width: 180,
        backgroundColor: "Background",
        color: "black",
        borderColor: "Background",
        fontFamily: "NanumSquareAceb",
        fontSize: 20,
        boxShadow: 0,
        displayPrint: "none",
      }}>
      <a
        href={teamGithubLink}
        target="_blank"
        rel="noreferrer"
        className="text-hyperlink"
        style={{ color: "black" }}>
        {teamGithubLinkDes}
      </a>
    </Fab>
  );
}

export default GithubLinkButton;
