import "./LandingPage.css";
import Grid from "@mui/material/Grid";
import ProjectDescription from "./components/ProjectDescription";

function LandingPage() {
  return (
    <Grid container>
      <Grid item xs className="grid-mainleft">
        <ProjectDescription></ProjectDescription>
      </Grid>
      <Grid item xs className="grid-mainright">
        <div className="background"></div>
      </Grid>
    </Grid>
  );
}
export default LandingPage;
