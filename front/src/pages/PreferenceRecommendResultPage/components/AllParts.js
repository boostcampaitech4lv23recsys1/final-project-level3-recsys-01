import React from "react";
import Grid from "@mui/material/Grid";
import AllCodiPartButton from "./AllCodiPartButton";

function AllParts({ fixPartList }) {
  const allPartsName = ["모자", "성형", "헤어", "상의", "하의", "신발", "무기"];
  console.log(fixPartList);
  console.log("aassssssaaa");
  const collectAllPart = () => {
    const all = [];
    for (let idx = 0; idx < allPartsName.length; idx++) {
      for (let id = 0; id < fixPartList.length; id++) {
        if (allPartsName[idx] === fixPartList[id]) {
          all.push(
            <Grid
              item
              xs={1}
              className="button-fixitem"
              key={allPartsName[idx]}>
              <AllCodiPartButton codiPart={fixPartList[id]}></AllCodiPartButton>
            </Grid>,
          );
          continue;
        }
        all.push(
          <Grid item xs={1} className="button-recitem" key={allPartsName[idx]}>
            <AllCodiPartButton codiPart={allPartsName[idx]}></AllCodiPartButton>
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
