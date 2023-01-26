import * as React from "react";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";

const top5Songs = [
  { label: "Organise" },
  { label: "Joha" },
  { label: "Terminator" },
  { label: "Dull" },
  { label: "Nzaza" },
];

function BasicSearch() {
  return (
    <Autocomplete
      disablePortal
      id="combo-box-demo"
      options={top5Songs}
      sx={{ width: 300 }}
      renderInput={(params) => <TextField {...params} label="Songs" />}
    />
  );
}

export default BasicSearch;
