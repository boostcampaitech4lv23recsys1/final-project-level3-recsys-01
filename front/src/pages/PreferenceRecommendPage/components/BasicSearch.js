import * as React from "react";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import { codiPartToData } from "./ItemLabel";

function BasicSearch(codiPart) {
  const codiPartKey = Object.values(codiPart)[0];
  const codiPartData = codiPartToData[codiPartKey];
  return (
    <Autocomplete
      id="popover-searchbox"
      options={codiPartData}
      sx={{ width: "300px" }}
      ListboxProps={{
        style: {
          maxHeight: "125px",
        },
      }}
      renderInput={(params) => <TextField {...params} label={codiPartKey} />}
    />
  );
}

export default BasicSearch;
