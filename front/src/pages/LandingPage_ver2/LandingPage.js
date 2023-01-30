import "./LandingPage.css";
import Grid from "@mui/material/Grid";
import ProjectDescription from "./components/ProjectDsecription";

function LandingPage() {
  return (
    <Grid container>
      <Grid item xs className="grid-mainleft">
        <ProjectDescription></ProjectDescription>
      </Grid>
      <Grid item xs className="grid-mainright">
        dd
      </Grid>
    </Grid>
  );
}
export default LandingPage;
