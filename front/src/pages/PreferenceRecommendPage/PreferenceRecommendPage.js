import "./PreferenceRecommendPage.css";
import Grid from "@mui/material/Grid";
import CodiPartInputs from "./components/CodiPartInputs";
import InfoTextAndVideo from "./components/InfoTextAndVideo";
import GoCodiRecResult from "./components/GoCodiRecResult";
import clickIcon from "../../assets/icons/click.png";
import CodiSimulator from "../../components/CodiSimulator";
import Stack from "@mui/material/Stack";

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
}) {
  const defaultFixObject = {
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  };
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
        />
      </Stack>
      <Grid
        container
        direction="row"
        justifyContent="center"
        alignItems="center">
        <GoCodiRecResult />
      </Grid>
    </div>
  );
}
export default PreferenceRecommendPage;
