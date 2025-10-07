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
2. 버튼을 누를 때마다 LSTM·랜덤 포레스트·회귀 기반의 알고리즘이 다섯 세트의 번호를 생성합니다.

### 로컬 서버에서 실행하고 싶을 때

```bash
python3 -m http.server 8080
# 브라우저에서 http://localhost:8080/index.html 접속
```

## 🔁 새로운 회차 반영 절차

1. `lotto.sql` 하단에 최신 회차의 `INSERT INTO lotto VALUES (...)` 문을 추가합니다.
2. 다음 명령을 실행해 `data/lotto_dataset.js`를 다시 만듭니다.
   ```bash
   python3 scripts/export_dataset.py
   ```
3. 새로 고침만 하면 웹페이지가 최신 데이터로 갱신됩니다.

> 참고: 변환 스크립트는 내장 `sqlite3` 모듈만 사용하므로 추가 패키지 설치가 필요 없습니다.

## ⚠️ 이용 안내

- 로또마스터는 과거 통계 기반 추천 도구일 뿐이며 실제 당첨을 보장하지 않습니다.
- 책임감 있는 게임 참여를 권장합니다.

행운을 빕니다! 🍀
