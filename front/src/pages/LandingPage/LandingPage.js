import "./LandingPage.css";
import TitleDescription from "./components/TitleDescription";
import TeamDescription from "./components/TeamDescription";
import ChoicePreferOrConcept from "./components/ChoicePreferOrConcept";

function LandingPage() {
  return (
    <div className="LandingPage">
      <TitleDescription></TitleDescription>
      <ChoicePreferOrConcept></ChoicePreferOrConcept>
      <TeamDescription></TeamDescription>
    </div>
  );
}
export default LandingPage;
