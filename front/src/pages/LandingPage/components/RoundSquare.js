import React from "react";
import Fab from "@mui/material/Fab";

function RoundSquare({ icon, description, movedescription }) {
  return (
    <div className="round-square">
      <center>
        <img
          alt=""
          src={icon}
          width="60px"
          height="60px"
          className="img-margin"
        />
        <p className="text-textinbutton">{description}</p>
        <Fab
          variant="extended"
          sx={{
            width: 250,
            backgroundColor: "#E5B8C8",
            color: "white",
            fontFamily: "NanumSquareAcb",
            fontSize: 20,
          }}>
          <a
            href="recommend/preference"
            className="text-hyperlink"
            style={{ color: "white" }}>
            {movedescription}
          </a>
        </Fab>
      </center>
    </div>
  );
}

export default RoundSquare;

// class RoundSquare extends React.Component {
//   constructor(description, icon) {
//     super();
//     this.description = description;
//     this.icon = icon;
//   }

//   render() {
//     return (
//       <div className="round-square">
//         <br></br>
//         <center>
//           <img alt="" src={this.icon} width="25px" height="25px" />
//           <p className="text-newline">{this.description}</p>
//           <Button variant="outlined">코디추천받기</Button>
//         </center>
//       </div>
//     );
//   }
// }

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
