import * as React from "react";
import { useTheme } from "@mui/material/styles";
import { useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import KeyboardArrowLeft from "@mui/icons-material/KeyboardArrowLeft";
import KeyboardArrowRight from "@mui/icons-material/KeyboardArrowRight";
import Paper from "@mui/material/Paper";
import { MobileStepper, Typography } from "@mui/material";
import SwipeableViews from "react-swipeable-views";
import { autoPlay } from "react-swipeable-views-utils";
import characterSSSU from "../../../assets/images/characterSSSU.png";
import characterJEONG from "../../../assets/images/characterJEONG.png";
import characterJANG from "../../../assets/images/characterJANG.png";
import characterRYU from "../../../assets/images/characterRYU.png";
import characterKIM from "../../../assets/images/characterKIM.png";

const AutoPlaySwipeableViews = autoPlay(SwipeableViews);
const images = [
  { label: "김은혜", imgSrc: characterKIM },
  { label: "류명현", imgSrc: characterRYU },
  { label: "이수경", imgSrc: characterSSSU },
  { label: "장원준", imgSrc: characterJANG },
  { label: "정준환", imgSrc: characterJEONG },
];

function SwipeableMemberStepper() {
  const theme = useTheme();
  const [activeStep, setActiveStep] = useState(0);
  const maxSteps = 5;

  const handleNext = () => {
    setActiveStep((preActiveStep) => preActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((preActiveStep) => preActiveStep - 1);
  };

  const handleStepChange = (step) => {
    setActiveStep(step);
  };

  return (
    <Box sx={{ maxWidth: 600, flexGrow: 1 }} className="memberBox-box">
      <MobileStepper
        className="memberBox-stepper"
        steps={maxSteps}
        position="static"
        activeStep={activeStep}
        sx={{ p: 2 }}
        nextButton={
          <Button
            size="small"
            onClick={handleNext}
            disabled={activeStep === maxSteps - 1}>
            {" "}
            Next
            {theme.direction === "rtl" ? (
              <KeyboardArrowLeft />
            ) : (
              <KeyboardArrowRight />
            )}
          </Button>
        }
        backButton={
          <Button size="small" onClick={handleBack} disabled={activeStep === 0}>
            {theme.direction === "rtl" ? (
              <KeyboardArrowRight />
            ) : (
              <KeyboardArrowLeft />
            )}
            Back
          </Button>
        }
      />
      <Paper
        className="memberBox-paper"
        square
        elevation={0}
        sx={{
          display: "flex",
          alignItems: "center",
          width: 600,
        }}>
        <AutoPlaySwipeableViews
          className="memberBox-autoPlay"
          axis={theme.direction == "rtl" ? "x-reverse" : "x"}
          index={activeStep}
          onChangeIndex={handleStepChange}
          enableMouseEvents>
          {images.map((step, index) => (
            <div key={step.label}>
              {Math.abs(activeStep - index) <= 2 ? (
                <Box
                  className="memberBox-img"
                  component="img"
                  sx={{
                    height: "100%",
                    maxWidth: 600,
                    display: "flex",
                    overflow: "hidden",
                    width: "100%",
                  }}
                  src={step.imgSrc}
                  alt={step.label}
                />
              ) : null}
            </div>
          ))}
        </AutoPlaySwipeableViews>
      </Paper>
    </Box>
  );
}

export default SwipeableMemberStepper;
