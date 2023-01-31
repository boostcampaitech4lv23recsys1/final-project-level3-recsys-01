import * as React from "react";
import Stack from "@mui/material/Stack";
import CodiPartButton from "../../../components/CodiPartButton";
import "./CodiPartInputs.css";

function CodiPartInputs() {
  return (
    <Stack direction="column" spacing={8} alignItems="center" sx={{ p: 8 }}>
      <Stack direction="row" spacing={6} alignItems="center">
        <CodiPartButton codiPart="모자" />
        <CodiPartButton codiPart="헤어" />
        <CodiPartButton codiPart="성형" />
      </Stack>
      <Stack direction="row" spacing={8} alignItems="center">
        <CodiPartButton codiPart="상의" />
        <CodiPartButton codiPart="하의" />
        <CodiPartButton codiPart="신발" />
        <CodiPartButton codiPart="무기" />
      </Stack>
    </Stack>
  );
}

export default CodiPartInputs;
