export default function TitleDescription({ title, description }) {
  return (
    <div className="text-defaultsetting">
      <h1>
        <center>{title}</center>
      </h1>
      <h3>
        <center>{description}</center>
      </h3>
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
