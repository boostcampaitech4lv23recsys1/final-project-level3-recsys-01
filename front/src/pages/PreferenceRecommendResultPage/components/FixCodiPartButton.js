import React from "react";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

function FixCodiPartButton({ codiPart }) {
  const returnFixItems = () => {
    const codiPrint = (
      <div className="fiximage">
        <Typography fontFamily={"NanumSquareAcr"} lineHeight="3">
          {codiPart[1]}
        </Typography>
        <img
          alt=""
          src={codiPart[0]["img"]}
          width="100"
          height="100"
          className="bgtop"
        />
        <Typography fontFamily={"NanumSquareAcr"}>
          {codiPart[0]["label"]}
        </Typography>
      </div>
    );
    return codiPrint;
  };
  return (
    <Stack direction="column" spacing={1} alignItems="center">
      {returnFixItems()}
    </Stack>
  );
}
export default FixCodiPartButton;
