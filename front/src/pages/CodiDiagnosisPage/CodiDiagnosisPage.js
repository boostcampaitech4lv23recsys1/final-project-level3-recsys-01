import "./CodiDiagnosisPage.css";
import CodiSimulator from "../../components/CodiSimulator";
import CodiPartInputs from "./components/CodiPartInputs";
import Grid from "@mui/material/Grid";
import GetDiagnosisResult from "./components/GetDiagnosisResult";

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
}) {
  return (
    <div className="CDP">
      <Grid container>
        {" "}
        <Grid item xs={3} paddingTop={"15%"} paddingLeft={"20%"}>
          <CodiSimulator
            inputHat={inputHat}
            inputHair={inputHair}
            inputFace={inputFace}
            inputTop={inputTop}
            inputBottom={inputBottom}
            inputShoes={inputShoes}
            inputWeapon={inputWeapon}
          />
        </Grid>
        <Grid item xs={9}>
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
          />
        </Grid>
      </Grid>
      <GetDiagnosisResult
        inputHat={inputHat}
        inputHair={inputHair}
        inputFace={inputFace}
        inputTop={inputTop}
        inputBottom={inputBottom}
        inputShoes={inputShoes}
        inputWeapon={inputWeapon}
      />
    </div>
  );
}

export default CodiDiagnosisPage;
