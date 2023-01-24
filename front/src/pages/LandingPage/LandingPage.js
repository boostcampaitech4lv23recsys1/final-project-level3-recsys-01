import PageTitle from "../../components/PageTitle";
import RoundSquare from "./components/RoundSquare";
import dinoEgg from "../../assets/icons/dino_egg_white.png";

function LandingPage() {
  const roundsquarepart = new RoundSquare(
    "부위별 코디 추천을 설명 \n 위 아이콘은 부위별 코디추천 아이콘",
    dinoEgg,
  );
  const title = "Home";
  return (
    <div>
      <h2>
        <PageTitle title={title}></PageTitle>
      </h2>
      <h3>"afds"</h3>
      {roundsquarepart.render()}
    </div>
  );
}
export default LandingPage;
