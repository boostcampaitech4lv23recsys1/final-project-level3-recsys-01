export default function PageTitle({ title }) {
  const pagedescription =
    "이 기능에 대해 뭐라고\n 한 두 줄 정도로 예쁘고 간결하게 설명하기";
  return (
    <div className="text-newline">
      <h1>
        <center>{title}</center>
      </h1>
      <h5>
        <center>{pagedescription}</center>
      </h5>
    </div>
  );
}

// 이건 왜 안됨?
// const PageTitle = ({ title }) => {
//   <h1>
//     <center>{title}</center>
//   </h1>;
// };

// export default PageTitle;
