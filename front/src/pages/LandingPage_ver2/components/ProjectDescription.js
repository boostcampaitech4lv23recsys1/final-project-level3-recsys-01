import * as React from "react";
import Slogan from "./Slogan";
import ProjectSummary from "./ProjectSummary";
import SubSummary from "./SubSummary";
import GoCodiRec from "./GoCodiRec";
import GoCodiDiagnosis from "./GoCodiDiagnosis";

function ProjectDescription() {
  return (
    <div className="text-defaultsetting">
      <Slogan></Slogan>
      <ProjectSummary></ProjectSummary>
      <SubSummary></SubSummary>
      <div className="button-gocodi">
        <GoCodiRec></GoCodiRec>
        <GoCodiDiagnosis></GoCodiDiagnosis>
      </div>
    </div>
  );
}

export default ProjectDescription;
