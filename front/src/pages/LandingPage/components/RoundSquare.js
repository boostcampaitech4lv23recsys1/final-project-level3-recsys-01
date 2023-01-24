import React from "react";
import dinoEgg from "../../../assets/icons/dino_egg_white.png";
import Button from "@mui/material/Button";

// function RoundSquare() {
//   return <div className="round-square">asdfadfs</div>;
// }

class RoundSquare extends React.Component {
  constructor(description, icon) {
    super();
    this.description = description;
    this.icon = icon;
  }

  render() {
    return (
      <div className="round-square">
        <center>
          <br></br>
          <img alt="" src={this.icon} width="25px" height="25px" />
          <br></br>
          <br></br>
          <div className="text-newline">{this.description}</div>
          <button>코디추천받기</button>
        </center>
      </div>
    );
  }
}
export default RoundSquare;
// import dinoEgg from "../../../assets/icons/dino_egg_white.png";

// class RoundSquare extends React.Component {
//   // React component를 정의하려면 React.Component를 상속받아야 함
//   constructor(description) {
//     // init 역할
//     this.icon = dinoEgg;
//     this.description = description;
//   }
//   getIcon() {
//     return this.icon;
//   }
//   getDescriptionText() {
//     return this.description;
//   }
//   render() {
//     const style = {
//       width: "100px",
//       height: "100px",
//     };
//     return <div style={style}></div>;
//   }
// }

// export default RoundSquare;

// class RoundSquare extends React.Component {
//   constructor(props) {
//     super(props);
//   }
//   render() {
//     const style = {
//       width: 100px,
//       height: 100,
//     };
//     return <div style={style}>{"adfs"}</div>;
//   }
// }

// export default RoundSquare;
