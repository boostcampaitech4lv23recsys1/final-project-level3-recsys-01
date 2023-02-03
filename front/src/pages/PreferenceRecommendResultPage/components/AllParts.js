import React from "react";
import Grid from "@mui/material/Grid";
import AllCodiPartButton from "./AllCodiPartButton";

function AllParts({ fixPartList, recommendData }) {
  const codiPartName = {
    Hat: "모자",
    Hair: "헤어",
    Face: "성형",
    Top: "상의",
    Bottom: "하의",
    Shoes: "신발",
    Weapon: "무기",
  };

  const collectAllPart = () => {
    const all = [];
    const codiPartEngName = Object.keys(codiPartName);
    for (let idx = 0; idx < codiPartEngName.length; idx++) {
      if (fixPartList.includes(codiPartEngName[idx])) {
        all.push(
          <Grid
            item
            xs={1}
            className="button-fixitem"
            key={codiPartEngName[idx]}>
            <AllCodiPartButton
              partName={codiPartName[codiPartEngName[idx]]}
              codiPart={
                recommendData[codiPartEngName[idx]]
              }></AllCodiPartButton>
          </Grid>,
        );
      } else {
        all.push(
          <Grid
            item
            xs={1}
            className="button-recitem"
            key={codiPartEngName[idx]}>
            <AllCodiPartButton
              partName={codiPartName[codiPartEngName[idx]]}
              codiPart={
                recommendData[codiPartEngName[idx]]
              }></AllCodiPartButton>
          </Grid>,
        );
      }
    }
    return all;
  };

  const buttonCollection = (
    <Grid container spacing={1} className="box-bestcodibox">
      <Grid item xs></Grid>
      {collectAllPart()}
      <Grid item xs></Grid>
    </Grid>
  );
  return buttonCollection;
}

export default AllParts;
