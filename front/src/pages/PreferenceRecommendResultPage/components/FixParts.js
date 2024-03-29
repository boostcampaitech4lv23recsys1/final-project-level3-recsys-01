import React from "react";
import FixCodiPartButton from "./FixCodiPartButton";
import Grid from "@mui/material/Grid";

function FixParts({ fixPartList }) {
  const collectFixPart = () => {
    const fixes = [];
    for (let idx = 0; idx < fixPartList.length; idx++) {
      fixes.push(
        <Grid item xs={1} key={fixPartList[idx]}>
          <FixCodiPartButton codiPart={fixPartList[idx]}></FixCodiPartButton>
        </Grid>,
      );
    }

    return fixes;
  };

  const buttonCollection = (
    <Grid container spacing={1} className="fiximage">
      <Grid item xs></Grid>
      {collectFixPart()}
      <Grid item xs></Grid>
    </Grid>
  );
  return buttonCollection;
}

export default FixParts;
