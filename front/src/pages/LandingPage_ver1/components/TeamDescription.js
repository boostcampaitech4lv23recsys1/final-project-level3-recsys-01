import github from "../../../assets/icons/github.png";
import GithubLinkButton from "../components/GithubLinkButton";

function TeamDescription() {
  const teamName = "Teams";
  const teamDes = "어떤 가치를 어쩌구\n 저쩌구\n 호롤롤루";
  return (
    <div className="text-defaultsetting">
      <h1>{teamName}</h1>
      <h3>{teamDes}</h3>
      <img
        alt=""
        src={github}
        width="50px"
        height="50px"
        className="img-github"
      />
      <GithubLinkButton className="body-two-margin"></GithubLinkButton>
    </div>
  );
}

export default TeamDescription;
