import React from "react";
import BestCodi from "./BestCodi";
import Grid from "@mui/material/Grid";

function BestCodiTopThree({ fixPartList }) {
  console.log("beset");
  console.log(fixPartList);
  return (
    <Grid>
      <Grid item xs={12}>
        <BestCodi order={1} fixPartList={fixPartList}></BestCodi>
      </Grid>
      <Grid item xs={12}>
        <BestCodi order={2} fixPartList={fixPartList}></BestCodi>
      </Grid>
      <Grid item xs={12}>
        <BestCodi order={3} fixPartList={fixPartList}></BestCodi>
      </Grid>
    </Grid>
  );
}

export default BestCodiTopThree;
