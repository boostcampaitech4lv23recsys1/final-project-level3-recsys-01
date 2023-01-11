### 첫 설정
```bash
apt update -y && apt upgrade -y
apt install curl

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash
source ~/.bashrc
nvm install 16.14.0
nvm use 16.14.0

npm install --global yarn
```

### 프론트엔드
- React (create-react-app으로 구현되었습니다.)
- React Bootstrap
- axios

### 프론트 엔드 서버 실행

```bash
cd front
yarn
yarn start
```