import React from "react";
import Box from '@mui/material/Box';
import AllParts from "./AllParts";

function BestCodi({ order, fixPartList }) {
    return (
        <div className="block-bestorder">
            <h2>Best {order}</h2>
            <AllParts fixPartList={fixPartList}></AllParts>
        </div>
    )
}

export default BestCodi