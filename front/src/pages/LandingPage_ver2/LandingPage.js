import "./LandingPage.css";
import Grid from "@mui/material/Grid";
import ProjectDescription from "./components/ProjectDescription";
import back from "../../assets/images/LandingBack.png";

function LandingPage() {
  return (
    <Grid container>
      <Grid item xs className="grid-mainleft">
        <ProjectDescription></ProjectDescription>
      </Grid>
      <Grid item xs className="grid-mainright">
        <div style={{ backgroundImage: { back } }} className="background"></div>
      </Grid>
    </Grid>
  );
}
export default LandingPage;
