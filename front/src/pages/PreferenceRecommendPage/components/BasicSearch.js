import * as React from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import { InputAdornment } from "@mui/material";

function BasicSearch({ codiPart, codiPartData, onSearchChange, inputValue }) {
  return (
    <>
      <Autocomplete
        id="popover-searchbox"
        options={codiPartData}
        sx={{ width: "300px" }}
        autoHighlight
        inputValue={inputValue}
        loading={true}
        onInputChange={(event, newInputValue) => {
          let newInputImage = event.target.children[0].src;
          let newInputId = event.target.id.split("-");
          newInputId = newInputId[3];
          onSearchChange(newInputValue, newInputImage, newInputId);
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
              alt={codiPartData.id}
            />
            {codiPartData.label}
          </Box>
        )}
        renderInput={(params) => <TextField {...params} label={codiPart} />}
      />
    </>
  );
}

export default BasicSearch;
