import "./App.css";
import * as Api from "./api";
// import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import Header from "./components/Header";
import Footer from "./components/Footer";
import About from "./pages/About/About";
import Contact from "./pages/Contact/Contact";
import LandingPage from "./pages/LandingPage_ver2/LandingPage"; // check Landing Page version
import PreferenceRecommendPage from "./pages/PreferenceRecommendPage/PreferenceRecommendPage";
import PreferenceRecommendResultPage from "./pages/PreferenceRecommendResultPage/PreferenceRecommendResultPage";

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" exact element={<Navigate to="/recommend" />} />
        <Route path="/recommend" exact element={<LandingPage />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route
          path="recommend/preference"
          element={<PreferenceRecommendPage />}
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
