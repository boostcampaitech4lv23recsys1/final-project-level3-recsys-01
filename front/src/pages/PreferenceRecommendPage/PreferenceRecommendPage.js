import "./PreferenceRecommendPage.css";
import CodiPartInputs from "./components/CodiPartInputs";
import InfoTextAndVideo from "./components/InfoTextAndVideo";

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
    </div>
  );
}
export default PreferenceRecommendPage;
