### Pycharm 에서 프로젝트별로 아나콘다 가상환경 사용

### 가상환경 생성 방법
1. Anaconda 설치
2. Anaconda Prompt 에서 가상환경 생성
``` 
conda create -n [환경이름] python=[버전]
```
3. 잘 설치되었는지 가상환경 리스트 확인
``` 
conda env list
```
4. 생성한 가상환경에서 필요한 패키지 다운로드
```
conda activate main
conda install pandas
```