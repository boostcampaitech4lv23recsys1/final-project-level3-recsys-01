import * as React from "react";
import { useNavigate } from "react-router-dom";

function GoCodiDiagnosis() {
  const navigate = useNavigate();
  return (
    <div>
      <button className="button-godig">
        <a
          href="/"
          onClick={(event) => {
            event.preventDefault();
            window.location.reload(navigate("diagnosis"));
          }}>
          {"코디 진단부터 받으러 가기"}
        </a>
      </button>
    </div>
  );
}

export default GoCodiDiagnosis;
