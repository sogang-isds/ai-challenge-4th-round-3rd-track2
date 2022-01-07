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

- 다운받은 모델파일과 샘플 데이터를 압축해제하여 `ai-challenge-4th-round-3rd-track2`디렉토리 밑에 위치시킵니다. (기존의 폴더를 덮어씌웁니다.)

- 최종적인 디렉토리 구성은 다음과 같습니다. 

  ![image](https://user-images.githubusercontent.com/86367674/148506812-af63da94-d2b9-41ee-9785-af2914f80d3f.png)


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

  

## 분류 

위협 상황은 아래와 같이 5가지의 클래스로 분류됩니다. 

- 020121 : 협박
- 000001 : 해당 없음 
- 02051 : 갈취공갈
- 020811 : 직장내 괴롭힘
- 020819 : 기타 괴롭힘 



## 실행 



### 텍스트 추론

- 아래와 같이 실행합니다.

  ```
  python predict.py
  ```



- 실행을 하면 아래와 같이 프롬프트가 나타나며, 여기에 샘플 텍스트를 입력합니다.

  ```
  INPUT> 강대리 네 과장님 부르셨습니까 강대리 이 머리 꼬라지가 뭐야 네 제 머리에 무슨 문제라도 있습니까 강대리 일을 못하면 단정하기라도 하던가 회사 출근할 때 머리도안감고 뭐하는거야 꼬질꼬질 해가지고는 신경 좀 써 과장님 요즘 강대리 집사람하고 사이가 안 좋다고 합니다 그럼 그렇지 집안도 제대로 건사하지 못하는 게 회사 일이라고 똑 부러지게 하겠냐 죄송합니다 오늘 아침부터 강대리 꼬라지 보니까 기분이 팍 상하네
  ```

  

- 결과로 클래스 분류코드와 폭력 분류명이 출력됩니다.

  ```
  020811, 직장내괴롭힘
  ```

  

### 오디오 추론

오디오 추론에서는 샘플 오디오파일을 분할 및 음성인식 후 텍스트 추론단계를 거쳐 결과를 출력합니다. 

- 아래와 같이 실행합니다.

  ```
  python predict_audio.py
  ```

  

- 실행을 하면 아래와 같이 프롬프트가 나타나며, 여기에 샘플 데이터 경로를 입력합니다. 

  ```
  파일 경로를 입력하세요.
  INPUT> sample_data/t2_001.wav
  ```

  

- 실행 결과는 아래와 같습니다. 

 ![predict_audio_demo](https://user-images.githubusercontent.com/86367674/148507050-7c71b068-9b57-43d8-a2d7-2d3eb491362d.gif)


  ```
  Audio splitting...
  splitted : ./tmp/chunk_000.wav, 0.00 - 1.19
  splitted : ./tmp/chunk_001.wav, 1.19 - 2.86
  splitted : ./tmp/chunk_002.wav, 2.86 - 4.03
  splitted : ./tmp/chunk_003.wav, 4.03 - 5.23
  splitted : ./tmp/chunk_004.wav, 5.23 - 6.14
  splitted : ./tmp/chunk_005.wav, 6.14 - 7.74
  splitted : ./tmp/chunk_006.wav, 7.74 - 8.56
  splitted : ./tmp/chunk_007.wav, 8.56 - 10.62
  splitted : ./tmp/chunk_008.wav, 10.62 - 11.77
  splitted : ./tmp/chunk_009.wav, 11.77 - 15.02
  splitted : ./tmp/chunk_010.wav, 15.02 - 18.18
  splitted : ./tmp/chunk_011.wav, 18.18 - 19.71
  splitted : ./tmp/chunk_012.wav, 19.71 - 21.98
  splitted : ./tmp/chunk_013.wav, 21.98 - 25.79
  splitted : ./tmp/chunk_014.wav, 25.79 - 31.92
  splitted : ./tmp/chunk_015.wav, 31.92 - 33.08
  splitted : ./tmp/chunk_016.wav, 33.08 - 38.69
  
  Speech recognizing...
  recognized :./tmp/chunk_000.wav, 
  recognized :./tmp/chunk_001.wav, 강 대리
  recognized :./tmp/chunk_002.wav, 예 과장님
  recognized :./tmp/chunk_003.wav, 불렀으니까
  recognized :./tmp/chunk_004.wav, 강 대리
  recognized :./tmp/chunk_005.wav, 이 머리 꼬라지가 뭐야
  recognized :./tmp/chunk_006.wav, 예
  recognized :./tmp/chunk_007.wav, 제 머리에 무슨 문제라도 있으니
  recognized :./tmp/chunk_008.wav, 강 대리
  recognized :./tmp/chunk_009.wav, 일을 못 하면 단정하게 하도 하든가
  recognized :./tmp/chunk_010.wav, 회사 출근할 때 머리도 안 감고 뭐 하는 거야
  recognized :./tmp/chunk_011.wav, 꼬질꼬질 해 가지고
  recognized :./tmp/chunk_012.wav, 신경 좀 써
  recognized :./tmp/chunk_013.wav, 요즘 강대리 집사람하고 사이가 안 좋다
  recognized :./tmp/chunk_014.wav, 그럼 그렇지 집안도 제대로 검사 하지 못하는게 회사 일이 너무 똑부러지게 하겠어
  recognized :./tmp/chunk_015.wav, 죄송합니다
  recognized :./tmp/chunk_016.wav, 오늘 아침부터 강대리 꼬라지 보니까 기분이 팍 상하네
  
  Predicting...
  020811, 직장내괴롭힘
  ```

  












