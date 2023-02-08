import React from "react";
import Grid from "@mui/material/Grid";
import basicitem from "../../../assets/images/basicItem.png";
import fixitme from "../../../assets/images/fixItem.png";
import simulatorBg from "../../../assets/images/simulatorBg.png";
import Typography from "@mui/material/Typography";
import CodiSimulator from "../../../components/CodiSimulator";

function AllParts({ fixPartList, recommendData }) {
  const codiPartName = {
    Hat: "모자",
    Hair: "헤어",
    Face: "성형",
    Top: "상의",
    Bottom: "하의",
    Shoes: "신발",
    Weapon: "무기",
  };
  const collectAllPart = () => {
    const all = [];
    const codiPartEngName = Object.keys(codiPartName);

    all.push(
      <Grid xs={1.3} className="button-fixitem" key="CodiSumulator">
        <div className="img-instack">
          <img alt="" src={simulatorBg} width="180" height="250"></img>
        </div>
        <div className="img-inbox">
          <CodiSimulator
            className="codiSimulator"
            inputHat={recommendData["Hat"]}
            inputHair={recommendData["Hair"]}
            inputFace={recommendData["Face"]}
            inputTop={recommendData["Top"]}
            inputBottom={recommendData["Bottom"]}
            inputShoes={recommendData["Shoes"]}
            inputWeapon={recommendData["Weapon"]}
            size={2}
            isResult={true}></CodiSimulator>
        </div>
      </Grid>,
    );

    for (let idx = 0; idx < codiPartEngName.length; idx++) {
      if (fixPartList.includes(codiPartEngName[idx])) {
        all.push(
          <Grid xs={1.3} className="button-fixitem" key={codiPartEngName[idx]}>
            <div className="img-instack">
              <img alt="" src={fixitme} width="180" height="250"></img>
            </div>
            <div className="img-inbox">
              <img
                alt=""
                src={recommendData[codiPartEngName[idx]]["gcs_image_url"]}
                width="70%"
                height="70%"
              />
            </div>
            <div className="text-inboxname">
              <Typography fontFamily={"NanumSquareAcr"}>
                {recommendData[codiPartEngName[idx]]["name"]}
              </Typography>
            </div>
            <div className="text-inboxpart">
              <Typography fontFamily={"NanumSquareAceb"} fontSize="large">
                {codiPartName[codiPartEngName[idx]]}
              </Typography>
            </div>
          </Grid>,
        );
      } else {
        all.push(
          <Grid xs={1.3} className="button-fixitem" key={codiPartEngName[idx]}>
            <div className="img-instack">
              <img alt="" src={basicitem} width="170" height="250"></img>
            </div>
            <div className="img-inbox">
              <img
                alt=""
                src={recommendData[codiPartEngName[idx]]["gcs_image_url"]}
                width="70%"
                height="70%"
              />
            </div>
            <div className="text-inboxname">
              <Typography fontFamily={"NanumSquareAcr"}>
                {recommendData[codiPartEngName[idx]]["name"]}
              </Typography>
            </div>
            <div className="text-inboxpart">
              <Typography fontFamily={"NanumSquareAceb"} fontSize="large">
                {codiPartName[codiPartEngName[idx]]}
              </Typography>
            </div>
          </Grid>,
        );
      }
    }
    return all;
  };

  const buttonCollection = (
    <Grid container spacing={1} className="box-bestcodibox">
      <Grid item xs></Grid>
      {collectAllPart()}
      <Grid item xs></Grid>
    </Grid>
  );
  return buttonCollection;
}

export default AllParts;
