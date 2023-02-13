import "./PreferenceRecommendPage.css";
import Grid from "@mui/material/Grid";
import CodiPartInputs from "./components/CodiPartInputs";
import InfoTextAndVideo from "./components/InfoTextAndVideo";
import GoCodiRecResult from "./components/GoCodiRecResult";
import GoReviewPage from "../../components/GoReviewPage";

import CodiSimulator from "../../components/CodiSimulator";
import Stack from "@mui/material/Stack";
import { useState } from "react";

function PreferenceRecommendPage({
  inputHat,
  setInputHat,
  inputHair,
  setInputHair,
  inputFace,
  setInputFace,
  inputTop,
  setInputTop,
  inputBottom,
  setInputBottom,
  inputShoes,
  setInputShoes,
  inputWeapon,
  setInputWeapon,
  numberState,
  setNumberState,
}) {
  return (
    <div className="PreferenceRecommendPage">
      <InfoTextAndVideo />
      <Stack className="simulatorAndItem" direction="row" spacing={20}>
        <CodiSimulator
          className="codiSimulator"
          inputHat={inputHat}
          inputHair={inputHair}
          inputFace={inputFace}
          inputTop={inputTop}
          inputBottom={inputBottom}
          inputShoes={inputShoes}
          inputWeapon={inputWeapon}
          size={4.5}
          isResult={false}
        />
        <CodiPartInputs
          inputHat={inputHat}
          setInputHat={setInputHat}
          inputHair={inputHair}
          setInputHair={setInputHair}
          inputFace={inputFace}
          setInputFace={setInputFace}
          inputTop={inputTop}
          setInputTop={setInputTop}
          inputBottom={inputBottom}
          setInputBottom={setInputBottom}
          inputShoes={inputShoes}
          setInputShoes={setInputShoes}
          inputWeapon={inputWeapon}
          setInputWeapon={setInputWeapon}
          numberState={numberState}
          setNumberState={setNumberState}
        />
      </Stack>
      <Grid
        container
        direction="column"
        justifyContent="center"
        alignItems="center">
        <GoCodiRecResult numberState={numberState} />
        <GoReviewPage />
      </Grid>
    </div>
  );
}
export default PreferenceRecommendPage;
