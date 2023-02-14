import * as React from "react";
import { useState } from "react";
import Fab from "@mui/material/Fab";
import { useNavigate } from "react-router-dom";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import Button from "@mui/material/Button";

function GoCodiRecResult({ numberState }) {
  const navigate = useNavigate();
  const [alertOpen, setAlertOpen] = useState(false);
  const handleAlertOpen = () => {
    setAlertOpen(true);
  };
  const handleAlertClose = () => {
    setAlertOpen(false);
  };
  return (
    <div>
      <Fab
        variant="extended"
        onClick={() => {
          if (numberState <= 2 || numberState >= 7) {
            handleAlertOpen();
          } else {
            navigate("result");
          }
        }}
        sx={{
          marginTop: 5,
          borderRadius: 3,
          border: 1,
          width: 500,
          height: 60,
          backgroundColor: "#8A37FF",
          color: "white",
          fontFamily: "NanumSquareAcb",
          fontSize: 30,
        }}>
        <a
          href="/"
          onClick={(event) => {
            event.preventDefault();
          }}
          style={{ color: "white" }}>
          {"코디 추천 받으러 가기"}
        </a>
      </Fab>
      <Dialog
        open={alertOpen}
        onClose={handleAlertClose}
        aria-describedby="alert-dialog-description">
        {numberState <= 2 ? (
          <DialogContent id="alert-dialog-description">
            <DialogContentText>
              최소 세 부위 이상 선택해주세요.
            </DialogContentText>
          </DialogContent>
        ) : (
          <DialogContent id="alert-dialog-description">
            <DialogContentText>
              모든 부위 선택 시 추천이 불가능합니다.
            </DialogContentText>
          </DialogContent>
        )}

        <DialogActions>
          <Button onClick={handleAlertClose} autoFocus>
            {" "}
            Close{" "}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}

export default GoCodiRecResult;
