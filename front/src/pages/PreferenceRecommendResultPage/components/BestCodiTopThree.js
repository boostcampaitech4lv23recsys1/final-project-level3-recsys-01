import React from "react";
import BestCodi from "./BestCodi";
import Grid from "@mui/material/Grid";

function BestCodiTopThree({ fixPartList, recommendData }) {
  return (
    <Grid>
      <Grid item xs={12}>
        <BestCodi
          order={1}
          fixPartList={fixPartList}
          recommendData={recommendData[0]}></BestCodi>
      </Grid>
      <Grid item xs={12}>
        <BestCodi
          order={2}
          fixPartList={fixPartList}
          recommendData={recommendData[1]}></BestCodi>
      </Grid>
      <Grid item xs={12}>
        <BestCodi
          order={3}
          fixPartList={fixPartList}
          recommendData={recommendData[2]}></BestCodi>
      </Grid>
    </Grid>
  );
}

export default BestCodiTopThree;
