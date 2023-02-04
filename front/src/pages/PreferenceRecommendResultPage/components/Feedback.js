import good from "../../../assets/images/happy.png";
import bad from "../../../assets/images/calm.png";
import check from "../../../assets/images/check.png";

import * as API from "../../../api";

function Feedback({ mode, setfeedback, disabled, setdisabled, recommendData }) {
  const { Hat, Hair, Face, Top, Bottom, Shoes, Weapon } = recommendData;
  let feedbackData = {
    Hat: {
      item_id: Hat.item_id,
      item_index: Hat.index,
    },
    Hair: {
      item_id: Hair.item_id,
      item_index: Hair.index,
    },
    Face: {
      item_id: Face.item_id,
      item_index: Face.index,
    },
    Top: {
      item_id: Top.item_id,
      item_index: Top.index,
    },
    Bottom: {
      item_id: Bottom.item_id,
      item_index: Bottom.index,
    },
    Shoes: {
      item_id: Shoes.item_id,
      item_index: Shoes.index,
    },
    Weapon: {
      item_id: Weapon.item_id,
      item_index: Weapon.index,
    },
  };

  const onClickFeedback = ({ goodbad }) => {
    if (goodbad === "good") {
      setfeedback(1);
      setdisabled(true);
      feedbackData["is_positive"] = true;
    } else if (goodbad === "bad") {
      setfeedback(0);
      setdisabled(true);
      feedbackData["is_positive"] = false;
    } else {
      setfeedback(-1);
    }
    sendFeedback(feedbackData);
  };

  const sendFeedback = async (feedbackData) => {
    await API.post("feedback", feedbackData);
  };

  // use state 쓰기

  return (
    <div className="button-goodbads">
      <button
        className="button-goodbad"
        onClick={() => {
          onClickFeedback({ goodbad: "good" });
        }}
        disabled={disabled}>
        {mode === 1 ? (
          <img alt="" src={check} width="20" height="20"></img>
        ) : (
          <img alt="" src={good} width="20" height="20"></img>
        )}
      </button>
      <button
        className="button-goodbad"
        onClick={() => onClickFeedback({ goodbad: "bad" })}
        disabled={disabled}>
        {mode === 0 ? (
          <img alt="" src={check} width="20" height="20"></img>
        ) : (
          <img alt="" src={bad} width="20" height="20"></img>
        )}
      </button>
    </div>
  );
}

export default Feedback;
