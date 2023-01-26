import * as React from "react";
import Stack from "@mui/material/Stack";
import CodiPartButton from "../../../components/CodiPartButton";
import "./CodiPartInputs.css";

function CodiPartInputs() {
  return (
    <Stack direction="column" spacing={8} alignItems="center" sx={{ p: 8 }}>
      <Stack direction="row" spacing={6} alignItems="center">
        <CodiPartButton codiPart="헤어" clickable={true} />
        <CodiPartButton codiPart="머리" clickable={true} />
        <CodiPartButton codiPart="성형" clickable={true} />
      </Stack>
      <Stack direction="row" spacing={8} alignItems="center">
        <CodiPartButton codiPart="상의" clickable={true} />
        <CodiPartButton codiPart="하의" clickable={true} />
        <CodiPartButton codiPart="신발" clickable={true} />
        <CodiPartButton codiPart="무기" clickable={false} />
      </Stack>
    </Stack>
  );
}

export default CodiPartInputs;
