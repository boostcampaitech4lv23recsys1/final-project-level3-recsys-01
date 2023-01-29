import React from "react";
import Grid from "@mui/material/Grid";
import FixCodiPartButton from "./FixCodiPartButton";


function AllParts({ fixPartList }) {
    const allPartsName = ['모자', '성형', '헤어', '상의', '하의', '신발', '무기']
    const collectAllPart = () => {
        const all = [];
        for (let idx = 0; idx < allPartsName.length; idx++) {
            if (fixPartList.includes(allPartsName[idx])) {
                all.push(
                    <Grid item xs={1} className="button-fixitem">

                        <FixCodiPartButton codiPart={allPartsName[idx]} >
                        </FixCodiPartButton>

                    </Grid>);
            } else {
                all.push(
                    <Grid item xs={1} className="button-recitem">
                        <FixCodiPartButton codiPart={allPartsName[idx]}>
                        </FixCodiPartButton>
                    </Grid>);
            }
        }
        return all;
    };

    const buttonCollection = (
        <Grid container spacing={1} className="box-bestcodibox">
            <Grid item xs></Grid>
            {collectAllPart()}
            {console.log(collectAllPart())}
            <Grid item xs></Grid>
        </Grid>
    )
    return (buttonCollection)
}

export default AllParts