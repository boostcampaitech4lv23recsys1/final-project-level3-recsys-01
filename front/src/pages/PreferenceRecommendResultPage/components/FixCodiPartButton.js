import React from "react";
import Fab from "@mui/material/Fab";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

function FixCodiPartButton({ codiPart }) {
    const itemImg = null
    return (
        <Stack direction="column" spacing={1} alignItems="center">
            <Typography>
                <b>{codiPart}</b>
            </Typography>
            <Fab aria-label="NotClickable">{itemImg}</Fab>
        </Stack>
    )
}
export default FixCodiPartButton;