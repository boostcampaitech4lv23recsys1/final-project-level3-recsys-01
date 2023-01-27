import React from "react";

function Member({ img, name, intro }) {
  return (
    <div>
      <img src={img} width="50px" height="50px" className="img-member"></img>
    </div>
  );
}
