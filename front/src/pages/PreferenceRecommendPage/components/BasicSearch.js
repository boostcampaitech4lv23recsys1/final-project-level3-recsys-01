import * as React from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";

function BasicSearch({
  codiPart,
  codiPartData,
  onSearchChange,
  inputValue,
  inputId,
  inputCategory,
}) {
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
          if (event && event.target.children[0]) {
            let newInputImage = event.target.children[0].src;
            let alt = event.target.children[0].alt.split(" ");
            let newInputId = alt[0];
            let newInputCategory = alt[1];
            onSearchChange(
              newInputValue,
              newInputImage,
              newInputId,
              newInputCategory,
            );
            console.log(newInputCategory);
          } else {
            onSearchChange(newInputValue);
          }
        }}
        getOptionLabel={(options) => options.label}
        renderOption={(props, options) => (
          <Box
            component="li"
            sx={{ "& > img": { mr: 2, flexShrink: 0 } }}
            {...props}>
            <img
              loading="lazy"
              width="20"
              src={options.img}
              srcSet={`${options.img} 2x`}
              alt={options.id + " " + options.category}
            />
            {options.label}
          </Box>
        )}
        renderInput={(params) => <TextField {...params} label={codiPart} />}
      />
    </>
  );
}

export default BasicSearch;
