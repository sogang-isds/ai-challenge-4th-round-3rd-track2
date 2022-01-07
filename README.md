# 인공지능 그랜드 챌린지 4차 대회 3단계 트랙2

그랜드 챌린드 4차대회 3단계에서 수행했던 모델을 공개합니다. 

API 형식으로 위협 상황을  인지할 수 있습니다. 



## Requirements 

- python => 3.8.12

- torch => 1.10.1+cu113

  

## Installation

- 아래의 명령을 이용해 소스코드를 받습니다.

  ```
  https://github.com/sogang-isds/ai-challenge-4th-round-3rd-track2.git
  ```

- 아래의 구글 드라이브 링크를 통해 모델파일과 샘플 데이터를 다운받습니다. 

  https://drive.google.com/drive/folders/1Bwnoqo1fG3A97Dig-HqlgOqF0T-xDlx_?usp=sharing

- 다운받은 모델파일과 샘플 데이터를 압축해제하여 `ai-challenge-4th-round-3rd-track2`디렉토리 밑에 위치시킵니다.

- 최종적인 디렉토리 구성은 다음과 같습니다. 

- <디렉토리 구성 완료된 사진>

- python 가상 환경을 설치합니다.

  ```
  cd ai-challenge-4th-round-3rd-track2
  virtualenv -p python3 myenv
  source myenv/bin/activate
  ```

- https://pytorch.org/ 에 접속하여 환경에 맞는 pytorch 설치 명령어를 확인한 뒤, 가상환경에 pytorch를 설치합니다.

  ```
  pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```

- python 필요한 패키지들을 설치합니다. 

  ```
  pip install -r requirements.txt
  ```

  
