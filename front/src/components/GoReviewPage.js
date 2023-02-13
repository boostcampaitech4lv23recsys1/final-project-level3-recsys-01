import * as React from "react";
import "./GoReviewPage.css";
import { Link } from "react-router-dom";

function GoReviewPage() {
  return (
    <div>
      <button className="button-goReview">
        <Link
          className="link-goReview"
          to="/recommend/review"
          target="_blank"
          rel="noopener noreferrer">
          {" "}
          메신사 서비스에 대한 리뷰를 남겨주세요!{" "}
        </Link>
      </button>
    </div>
  );
}

export default GoReviewPage;

{
  /* <a
onClick={() => (window.location.href = "/recommend/review")}
target="_blank"
rel="noreferrer">
메신사 서비스에 대한 리뷰를 남겨주세요!
</a> */
}
