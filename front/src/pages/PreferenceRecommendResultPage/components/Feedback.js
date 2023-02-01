import good from "../../../assets/images/happy.png";
import bad from "../../../assets/images/calm.png";
import check from "../../../assets/images/check.png";

function Feedback({ mode, setfeedback }) {
  const onClickFeedback = ({ goodbad }) => {
    if (goodbad === "good") {
      setfeedback(1);
    } else if (goodbad === "bad") {
      setfeedback(0);
    } else {
      setfeedback(-1);
    }
  };

  // use state 쓰기

  return (
    <div className="button-goodbads">
      <button
        className="button-goodbad"
        onClick={() => {
          onClickFeedback({ goodbad: "good" });
          onClickDisabled();
        }}
        disabled={onClickDisabled({ mode })}>
        {mode === 1 ? (
          <img alt="" src={check} width="20" height="20"></img>
        ) : (
          <img alt="" src={good} width="20" height="20"></img>
        )}
        {console.log(mode)}
      </button>
      <button
        className="button-goodbad"
        onClick={() => onClickFeedback({ goodbad: "bad" })}>
        {mode === 0 ? (
          <img alt="" src={check} width="20" height="20"></img>
        ) : (
          <img alt="" src={bad} width="20" height="20"></img>
        )}
        {console.log(mode)}
      </button>
    </div>
  );
}

export default Feedback;
