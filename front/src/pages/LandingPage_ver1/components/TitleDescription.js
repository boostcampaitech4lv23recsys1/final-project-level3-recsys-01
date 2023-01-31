function TitleDescription() {
  const mainTitle = "Home";
  const mainTitleDes =
    "이 기능에 대해 뭐라고\n 한 두 줄 정도로 예쁘고 간결하게 설명하기";
  return (
    <div className="text-defaultsetting">
      <h1>{mainTitle}</h1>
      <h3>{mainTitleDes}</h3>
    </div>
  );
}
export default TitleDescription;
