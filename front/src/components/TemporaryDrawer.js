import * as React from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Button from "@mui/material/Button";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import menu from "../assets/icons/menuicon.png";
import reviewIcon from "../assets/icons/reviewicon.png";
import recIcon from "../assets/icons/recicon.png";
import dignosisIcon from "../assets/icons/dignosisicon.png";

import { useNavigate } from "react-router-dom";

function TemporaryDrawer() {
  const navigate = useNavigate();
  const [state, setState] = React.useState({
    right: false,
  });

  const toggleDrawer = (anchor, open) => (event) => {
    if (
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }

    setState({ ...state, [anchor]: open });
  };

  const iconList = [recIcon, dignosisIcon, reviewIcon];
  const pageList = ["preference/", "diagnosis", "review"];
  const infoList = ["recommend", "about"];

  const list = (anchor) => (
    <Box
      sx={{ width: anchor === "top" || anchor === "bottom" ? "auto" : 250 }}
      role="presentation"
      onClick={toggleDrawer(anchor, false)}
      onKeyDown={toggleDrawer(anchor, false)}>
      <List>
        {["코디 추천", "코디 진단", "리뷰 남기기"].map((text, index) => (
          <a
            href="/"
            onClick={(event) => {
              event.preventDefault();
              window.location.reload(navigate("recommend/" + pageList[index]));
            }}
            key={text}
            style={{
              color: "black",
            }}>
            <ListItem disablePadding>
              <ListItemButton>
                <ListItemIcon>
                  <img
                    src={iconList[index]}
                    alt=""
                    width="30"
                    height="30"></img>
                </ListItemIcon>
                <ListItemText primary={text} />
              </ListItemButton>
            </ListItem>
          </a>
        ))}
      </List>
      <Divider />
      <List>
        {["Home", "About"].map((text, index) => (
          <a
            href="/"
            onClick={(event) => {
              event.preventDefault();
              window.location.reload(navigate(infoList[index]));
            }}
            key={text}
            style={{
              color: "black",
            }}>
            <ListItem alignItems="flex-start" disablePadding>
              <ListItemButton>
                <ListItemText primary={text} />
              </ListItemButton>
            </ListItem>
          </a>
        ))}
      </List>
    </Box>
  );

  return (
    <div>
      {["right"].map((anchor) => (
        <React.Fragment key={anchor}>
          <Button onClick={toggleDrawer(anchor, true)}>
            <img
              alt=""
              src={menu}
              width="30"
              height="30"
              className="button-menu"></img>
          </Button>
          <Drawer
            anchor={anchor}
            open={state[anchor]}
            onClose={toggleDrawer(anchor, false)}>
            {list(anchor)}
          </Drawer>
        </React.Fragment>
      ))}
    </div>
  );
}

export default TemporaryDrawer;
