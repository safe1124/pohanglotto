# 🍀 로또마스터 (Lotto Master)

애플 공식 웹사이트 느낌의 단일 페이지로 한국 로또(6/45) 추천 조합을 보여주는 프로젝트입니다. 
백엔드는 필요하지 않으며, 순수 HTML/CSS/JavaScript만으로 다섯 가지 추천 세트를 생성합니다.

## 📁 폴더 구조

```
lotto-main/
├── index.html           # 로또마스터 웹페이지 (브라우저로 바로 실행)
├── data/
│   └── lotto_dataset.js # 회차 데이터(자동 생성 파일)
├── lotto.sql            # 공식 당첨 내역을 INSERT 구문으로 정리한 소스
└── scripts/
    └── export_dataset.py# lotto.sql을 JS 데이터로 변환하는 스크립트
```

## 🚀 사용 방법

1. `index.html`을 더블 클릭하거나 정적 서버에서 열면 바로 예측을 체험할 수 있습니다.
2. 버튼을 누를 때마다 LSTM(최근 12·24·36회)과 랜덤 포레스트 기반 알고리즘이 다섯 세트의 번호를 생성합니다.

### TensorFlow LSTM 비교 실험

회차별 번호 데이터를 CSV(`draw_no,n1..n6`)로 준비한 뒤 아래 명령으로 12·24·36회 창을 각각 학습/예측할 수 있습니다.

```bash
pip install tensorflow pandas numpy
python3 scripts/train_lstm_models.py --csv lotto.csv
```

출력은 창 길이별로 가장 확률이 높은 6개 번호 조합을 제공합니다.

### 로컬 서버에서 실행하고 싶을 때

```bash
python3 -m http.server 8080
# 브라우저에서 http://localhost:8080/index.html 접속
```

## 🔁 새로운 회차 데이터 추가하기

### 📝 단계별 가이드

#### 1단계: SQL 파일에 최신 회차 추가
`lotto.sql` 파일의 **맨 위쪽**(13번째 줄)에 최신 회차 데이터를 추가합니다.

```sql
INSERT INTO lotto VALUES (1194, 3, 13, 15, 24, 33, 37, 2, NULL);
```

**형식:** `(회차번호, 번호1, 번호2, 번호3, 번호4, 번호5, 번호6, 보너스번호, NULL)`

#### 2단계: 데이터셋 파일 생성
터미널에서 아래 명령어를 실행하여 `data/lotto_dataset.js` 파일을 업데이트합니다.

```bash
python3 scripts/export_dataset.py
```

✅ **완료!** 이제 `index.html`을 새로고침하면 최신 회차가 반영된 예측을 확인할 수 있습니다.

> 💡 **팁:** 변환 스크립트는 Python 내장 `sqlite3` 모듈만 사용하므로 별도의 패키지 설치가 필요 없습니다.

## ⚠️ 이용 안내

- 로또마스터는 과거 통계 기반 추천 도구일 뿐이며 실제 당첨을 보장하지 않습니다.
- 책임감 있는 게임 참여를 권장합니다.

행운을 빕니다! 🍀
