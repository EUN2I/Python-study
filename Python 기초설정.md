## 1.  Pycharm 에서 프로젝트별로 아나콘다 가상환경 사용
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

## 2. 패키지, 모듈, 함수 차이 

- 패키지(Package) : 패키지는 관련된 모듈들을 그룹화하여 함께 제공
- 모듈(Module) : 모듈은 패키지의 구성 요소로, 단일 파이썬 파일에 코드가 저장된 형태. 모듈은 특정 기능 또는 작업을 수행하기 위한 함수, 클래스, 변수 등을 포함
- 함수(Function) : 함수는 재사용 가능한 코드 블록으로, 특정 작업을 수행하기 위해 설계
- scipy는 패키지, stats는 모듈, ttest는 함수