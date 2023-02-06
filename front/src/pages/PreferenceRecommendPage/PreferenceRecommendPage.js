import "./PreferenceRecommendPage.css";
import Grid from "@mui/material/Grid";
import CodiPartInputs from "./components/CodiPartInputs";
import InfoTextAndVideo from "./components/InfoTextAndVideo";
import GoCodiRecResult from "./components/GoCodiRecResult";
import clickIcon from "../../assets/icons/click.png";

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
      <button
        className="button-reload"
        style={{
          borderRadius: 30,
          width: 100,
          height: 30,
          border: 1,
          backgroundColor: "#b9b9b9",
          color: "white",
          fontFamily: "NanumSquareAcb",
          fontSize: 20,
        }}
        onClick={() => {
          setInputHat(defaultFixObject);
          setInputHair(defaultFixObject);
          setInputFace(defaultFixObject);
          setInputTop(defaultFixObject);
          setInputBottom(defaultFixObject);
          setInputShoes(defaultFixObject);
          setInputWeapon(defaultFixObject);
        }}>
        Reset
      </button>
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
