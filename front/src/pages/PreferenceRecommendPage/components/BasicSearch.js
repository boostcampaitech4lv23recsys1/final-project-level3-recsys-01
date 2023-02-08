import * as React from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import CircularProgress from "@mui/material/CircularProgress";

import { useState, useEffect } from "react";

import * as API from "../../../api";

const codiPartToCategory = {
  모자: "Hat",
  헤어: "Hair",
  성형: "Face",
  상의: "Top",
  하의: "Bottom",
  신발: "Shoes",
  무기: "Weapon",
};

function BasicSearch({ codiPart, onSearchChange, inputValue, setAnchorEl }) {
  const [open, setOpen] = useState(false);
  const [codiPartData, setCodiPartData] = useState([]);
  const loading = open && codiPartData.length === 0;
  const getCodiPartData = async (active) => {
    const codiPartCategory = codiPartToCategory[codiPart];
    if (!codiPart || !codiPartCategory) return;
    try {
      const res = await API.get(`items/${codiPartCategory}`);
      const data = res.data.items;
      if (active) {
        setCodiPartData(
          data.map((currentItem) => {
            return {
              label: currentItem["name"],
              img: currentItem["gcs_image_url"],
              id: currentItem["item_id"],
              category: currentItem["equip_category"],
              index: currentItem["index"],
            };
          }),
        );
      }
    } catch (err) {
      console.error(err);
    }
  };
  useEffect(() => {
    let active = true;
    if (!loading) {
      return undefined;
    }
    getCodiPartData(active);
    return () => {
      active = false;
    };
  });

  useEffect(() => {
    if (!open) {
      setCodiPartData([]);
    }
  }, [open]);

  return (
    <Autocomplete
      id="asynchronous-demo"
      sx={{ width: 300 }}
      open={open}
      onOpen={() => {
        setOpen(true);
      }}
      onClose={() => {
        setOpen(false);
      }}
      options={codiPartData}
      loading={loading}
      onInputChange={(event, newInputValue) => {
        let newInputImage = event.target.children[0].src;
        let newInputAlt = event.target.children[0].alt.split(" ");
        let [newInputId, newInputCategory, newInputIndex] = newInputAlt;
        let updatedInputValue = {
          label: newInputValue,
          img: newInputImage,
          id: newInputId,
          category: newInputCategory,
          index: newInputIndex,
        };
        onSearchChange(updatedInputValue);
        setAnchorEl(null);
      }}
      renderOption={(props, codiPartData) => (
        <Box
          component="li"
          sx={{ "& > img": { mr: 2, flexShrink: 0 } }}
          {...props}>
          <img
            loading="lazy"
            width="20"
            src={codiPartData.img}
            srcSet={`${codiPartData.img} 2x`}
            alt={
              codiPartData.id +
              " " +
              codiPartData.category +
              " " +
              codiPartData.index
            }
          />
          {codiPartData.label}
        </Box>
      )}
      renderInput={(params) => (
        <TextField
          {...params}
          label={codiPart}
          InputProps={{
            ...params.InputProps,
            endAdornment: (
              <React.Fragment>
                {loading ? (
                  <CircularProgress color="inherit" size={20} />
                ) : null}
                {params.InputProps.endAdornment}
              </React.Fragment>
            ),
          }}
        />
      )}
    />
  );
}

export default BasicSearch;
