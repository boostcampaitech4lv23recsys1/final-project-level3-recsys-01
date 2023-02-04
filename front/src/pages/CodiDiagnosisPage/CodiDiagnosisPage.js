import "./CodiDiagnosisPage.css";
import CodiSimulator from "../../components/CodiSimulator";
import CodiPartInputs from "./components/CodiPartInputs";
import Stack from "@mui/material/Stack";
import GetDiagnosisResult from "./components/GetDiagnosisResult";
import basicItem from "../../assets/images/basicItem.png";

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
        />
      </Stack>
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
