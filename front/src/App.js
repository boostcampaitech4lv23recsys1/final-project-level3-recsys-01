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
import LandingPage from "./pages/LandingPage_ver2/LandingPage"; // check Landing Page version
import PreferenceRecommendPage from "./pages/PreferenceRecommendPage/PreferenceRecommendPage";
import PreferenceRecommendResultPage from "./pages/PreferenceRecommendResultPage/PreferenceRecommendResultPage";

function App() {
  const [inputHat, setInputHat] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });
  const [inputHair, setInputHair] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });
  const [inputFace, setInputFace] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });
  const [inputTop, setInputTop] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });

  const [inputBottom, setInputBottom] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });
  const [inputShoes, setInputShoes] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });
  const [inputWeapon, setInputWeapon] = useState({
    label: "",
    img: clickIcon,
    id: "",
    category: "",
  });
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" exact element={<Navigate to="/recommend" />} />
        <Route path="/recommend" exact element={<LandingPage />} />
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
        <Route
          path="recommend/preference/result"
          element={<PreferenceRecommendResultPage />}
        />
      </Routes>
      <Footer />
    </Router>
  );
}

export default App;
