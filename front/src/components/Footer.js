// import AppBar from "@mui/material/AppBar"; // 페이지 꽉차게 bar 생성
// import Toolbar from "@mui/material/Toolbar"; // 글씨를 벽에서 떼 줌, 요소끼리 잘 붙여줌
// import Typography from "@mui/material/Typography";
// import { ThemeProvider, createTheme } from "@mui/material/styles"; // them 추가 및 생성

function Footer() {
  const footerdescription =
    "Contact | gongryongal1@gmil.com \n 네이버 부스트캠프 AI Tech 4기 RecSys Track 공룡알 팀의 최종 프로젝트입니다.";
  return <div className="footer">{footerdescription}</div>;
}

export default Footer;

// const footerTheme = createTheme({
//   palette: {
//     primary: {
//       main: "#FCCEAD",
//     },
//   },
//   typography: {
//     fontFamily: ["NanumSquareAceb", "sans-serif"].join(","),
//   },
// });

// function Footer() {
//   return (
//     <div className="wrap-container">
//       <ThemeProvider theme={footerTheme}>
//         <AppBar position="static" color="primary">
//           <Toolbar variant="dense">
//             <Typography variant="h6" color="white" component="div">
//               메이플스토리 코디 추천
//             </Typography>
//           </Toolbar>
//         </AppBar>
//       </ThemeProvider>
//     </div>
//   );
// }
