import React from "react";
import FixCodiPartButton from "./FixCodiPartButton";
import Grid from "@mui/material/Grid";

function FixParts({ fixPartList }) {
  console.log("fixpartsssssssssssssssss");
  console.log(fixPartList);
  const collectFixPart = () => {
    const fixes = [];
    for (let idx = 0; idx < fixPartList.length; idx++) {
      fixes.push(
        <Grid item xs={1} className="button-fixparts">
          <FixCodiPartButton codiPart={fixPartList[idx]}></FixCodiPartButton>
        </Grid>,
      );
    }

    return fixes;
  };

  const buttonCollection = (
    <Grid container spacing={1}>
      <Grid item xs></Grid>
      {collectFixPart()}
      <Grid item xs></Grid>
    </Grid>
  );
  return buttonCollection;
}

export default FixParts;
