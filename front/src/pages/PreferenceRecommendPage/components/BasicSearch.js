import * as React from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import { codiPartToCategory } from "./ItemLabel";
import * as API from "../../../api";
import { render } from "react-dom";

function handleInputChange(value) {
  console.log(value);
}

function BasicSearch(codiPart) {
  const [codiPartData, setCodiPartData] = React.useState([]);
  React.useEffect(() => {
    const getCodiPartData = async () => {
      const codiPartKey = Object.values(codiPart)[0];
      const codiPartCategory = codiPartToCategory[codiPartKey];
      if (!codiPart || !codiPartCategory) return;
      const res = await API.get(`items/${codiPartCategory}`);
      const data = res.data.items;
      const codiPartData = [];
      for (let currentItem of Object.values(data)) {
        codiPartData.push({
          label: currentItem["name"],
          img: currentItem["gcs_image_url"],
        });
      }
      setCodiPartData(codiPartData);
    };
    getCodiPartData();
  }, [codiPart, codiPartToCategory]);
  const codiPartKey = Object.values(codiPart)[0];
  return (
    <>
      <Autocomplete
        id="popover-searchbox"
        options={codiPartData}
        sx={{ width: "300px" }}
        autoHighlight
        onInputChange={handleInputChange}
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
              alt={codiPartData.label}
            />
            {codiPartData.label}
          </Box>
        )}
        renderInput={(params) => <TextField {...params} label={codiPartKey} />}
      />
    </>
  );
}

export default BasicSearch;
