import "./App.css";
import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import clickIcon from "./assets/icons/click.png";
import Header from "./components/Header";
import Footer from "./components/Footer";
import About from "./pages/About/About";
import LandingPage from "./pages/LandingPage_ver2/LandingPage"; // check Landing Page version
import PreferenceRecommendPage from "./pages/PreferenceRecommendPage/PreferenceRecommendPage";
import PreferenceRecommendResultPage from "./pages/PreferenceRecommendResultPage/PreferenceRecommendResultPage";
import CodiDiagnosisPage from "./pages/CodiDiagnosisPage/CodiDiagnosisPage";
import Review from "./pages/Review/Review";

function App() {
  const [inputHat, setInputHat] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });
  const [inputHair, setInputHair] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });
  const [inputFace, setInputFace] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });
  const [inputTop, setInputTop] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });

  const [inputBottom, setInputBottom] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });
  const [inputShoes, setInputShoes] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });
  const [inputWeapon, setInputWeapon] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
    index: "",
  });

  const [numberState, setNumberState] = useState(0);
  if (numberState < 0) {
    setNumberState(0);
  }

  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" exact element={<Navigate to="/recommend" />} />
        <Route path="/recommend" exact element={<LandingPage />} />
        <Route path="/about" element={<About />} />
        <Route path="/recommend/review" element={<Review />} />
        <Route
          path="recommend/diagnosis"
          element={
            <CodiDiagnosisPage
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
          }
        />
        <Route
          path="recommend/preference"
          element={
            <PreferenceRecommendPage
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
          }
        />
        <Route
          path="recommend/preference/result"
          element={
            <PreferenceRecommendResultPage
              inputHat={inputHat}
              inputHair={inputHair}
              inputFace={inputFace}
              inputTop={inputTop}
              inputBottom={inputBottom}
              inputShoes={inputShoes}
              inputWeapon={inputWeapon}
            />
          }
        />
      </Routes>

      <Footer />
    </Router>
  );
}

export default App;
