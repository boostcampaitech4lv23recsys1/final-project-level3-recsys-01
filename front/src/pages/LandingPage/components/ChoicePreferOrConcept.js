import React from "react";
import Grid from "@mui/material/Grid";
import RecOptionSquare from "../components/RecOptionSquare";

function ChoicePreferOrConcept() {
  const recOptionPrefer =
    "부위별 코디 추천을 설명\n 위 아이콘은 부위별 코디 추천 아이콘\n 취향 코디 추천입니다.";
  const recOptionConcept =
    "부위별 코디 추천을 설명\n 위 아이콘은 부위별 코디 추천 아이콘\n 컨셉 코디 추천입니다.";
  const recOptionButton = "코디 추천 받기";
  return (
    <Grid container spacing={1}>
      <Grid item xs></Grid>
      <Grid item xs={3} className="grid-center">
        <RecOptionSquare
          optiondes={recOptionPrefer}
          optionbutton={recOptionButton}></RecOptionSquare>
      </Grid>
      <Grid item xs={3} className="grid-center">
        <RecOptionSquare
          optiondes={recOptionConcept}
          optionbutton={recOptionButton}></RecOptionSquare>
      </Grid>
      <Grid item xs></Grid>
    </Grid>
  );
}

export default ChoicePreferOrConcept;
