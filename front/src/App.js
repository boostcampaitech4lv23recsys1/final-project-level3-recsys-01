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
import Contact from "./pages/Contact/Contact";
import LandingPage from "./pages/LandingPage_ver2/LandingPage"; // check Landing Page version
import PreferenceRecommendPage from "./pages/PreferenceRecommendPage/PreferenceRecommendPage";
import PreferenceRecommendResultPage from "./pages/PreferenceRecommendResultPage/PreferenceRecommendResultPage";
import CodiDiagnosisPage from "./pages/CodiDiagnosisPage/CodiDiagnosisPage";

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
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" exact element={<Navigate to="/recommend" />} />
        <Route path="/recommend" exact element={<LandingPage />} />
        <Route path="/about" element={<About />} />
        {/* <Route path="/contact" element={<Contact />} /> */}
        <Route
          path="/recommend/diagnosis"
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
            />
          }
        />
        {console.log(inputHat)}
        {console.log(inputHair)}
        {console.log(inputFace)}
        {console.log(inputTop)}
        {console.log(inputBottom)}
        {console.log(inputShoes)}
        {console.log(inputWeapon)}
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
