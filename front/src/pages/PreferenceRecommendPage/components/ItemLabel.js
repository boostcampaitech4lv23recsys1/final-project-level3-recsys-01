import * as React from "react";
import { useState, useEffect } from "react";
import * as API from "../../../api";

const codiPartToCategory = {
  모자: "Hat",
  헤어: "Hair",
  성형: "Face",
  한벌옷: "Overall",
  상의: "Top",
  하의: "Bottom",
  신발: "Shoes",
  무기: "Weapon",
};

function ItemGetFromDB(codiPart) {
  const [codiPartData, setCodiPartData] = useState([]);
  useEffect(() => {
    const getCodiPartData = async () => {
      const codiPartCategory = codiPartToCategory[codiPart];
      if (!codiPart || !codiPartCategory) return;
      const res = await API.get(`items/${codiPartCategory}`);
      const data = res.data.items;
      const codiPartData = [];
      for (let currentItem of Object.values(data)) {
        codiPartData.push({
          label: currentItem["name"],
          img: currentItem["gcs_image_url"],
          id: currentItem["item_id"],
        });
      }
      setCodiPartData(codiPartData);
    };
    getCodiPartData();
  }, [codiPart, codiPartToCategory]);

  return codiPartData;
}

export { ItemGetFromDB };
