import React from "react";
import Fab from "@mui/material/Fab";

function LinkButton({ link, displaylink }) {
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
      }}>
      <a
        href={link}
        target="_blank"
        rel="noreferrer"
        className="text-hyperlink"
        style={{ color: "black" }}>
        {displaylink}
      </a>
    </Fab>
  );
}

export default LinkButton;
