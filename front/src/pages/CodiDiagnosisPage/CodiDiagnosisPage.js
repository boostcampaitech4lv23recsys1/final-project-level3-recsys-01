import "./CodiDiagnosisPage.css";
import * as React from "react";
import CodiSimulator from "../../components/CodiSimulator";
import CodiPartInputs from "./components/CodiPartInputs";
import Stack from "@mui/material/Stack";
import GetDiagnosisResult from "./components/GetDiagnosisResult";
import ShowDiagnosisResult from "./components/ShowDiagnosisResult";
import { useState } from "react";
import InfoText from "./components/InfoText";
import GoReviewPage from "../../components/GoReviewPage";

function CodiDiagnosisPage({
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
  const [diagnosisScore, setDiagnosisScore] = useState(0);
  return (
    <div className="CDP">
      <InfoText></InfoText>
      <Stack className="simulatorAndItem" direction="row" spacing={20}>
        {" "}
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
          className="item"
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
      {diagnosisScore !== 0 ? (
        <ShowDiagnosisResult diagnosisScore={diagnosisScore} />
      ) : (
        <p></p>
      )}
      <GetDiagnosisResult
        inputHat={inputHat}
        inputHair={inputHair}
        inputFace={inputFace}
        inputTop={inputTop}
        inputBottom={inputBottom}
        inputShoes={inputShoes}
        inputWeapon={inputWeapon}
        diagnosisScore={diagnosisScore}
        setDiagnosisScore={setDiagnosisScore}
        numberState={numberState}
      />
      <GoReviewPage />
    </div>
  );
}

export default CodiDiagnosisPage;
