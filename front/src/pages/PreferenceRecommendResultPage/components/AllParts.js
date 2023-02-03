import React from "react";
import Grid from "@mui/material/Grid";
import FixCodiPartButton from "./FixCodiPartButton";
import basicItem from "../../../assets/images/basicItem.png";
import fixItem from "../../../assets/images/fixItem.png";

function AllParts({ fixPartList }) {
  const allPartsName = ["모자", "성형", "헤어", "상의", "하의", "신발", "무기"];
  const collectAllPart = () => {
    const all = [];
    for (let idx = 0; idx < allPartsName.length; idx++) {
      if (fixPartList.includes(allPartsName[idx])) {
        all.push(
          <Grid item xs={1} className="button-fixitem" key={allPartsName[idx]}>
            <FixCodiPartButton
              codiPart={allPartsName[idx]}
              bgImage={fixItem}></FixCodiPartButton>
          </Grid>,
        );
      } else {
        all.push(
          <Grid item xs={1} className="button-recitem" key={allPartsName[idx]}>
            <FixCodiPartButton
              codiPart={allPartsName[idx]}
              bgImage={basicItem}></FixCodiPartButton>
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
