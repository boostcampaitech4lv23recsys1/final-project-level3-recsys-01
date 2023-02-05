import * as React from "react";
import "./Review.css";

function Review() {
  return (
    <div className="RP">
      <iframe
        className="googleReview"
        src="https://docs.google.com/forms/d/e/1FAIpQLSeaBu1ix3gR5ClQk9ZFgo_LaruEBHOPzC7ngVGnRnzPCT4UbA/viewform?embedded=true"
        width="700"
        height="1200"
        frameborder="0"
        marginheight="0"
        marginwidth="0">
        로드 중…
      </iframe>
    </div>
  );
}

export default Review;
