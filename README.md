# 🧠 RAG-PDF-QA with Local DeepSeek R1

이 프로젝트는 LangChain과 Ollama 기반의 LLM(DeepSeek R1), 임베딩 모델(mxbai-embed-large), Chroma 벡터 저장소를 활용하여 PDF 문서에 대한 질문에 답변하는 RAG기반 챗봇 시스템입니다.

로컬 환경에서 RAG 관련 프로젝트를 구축해보고 싶다는 생각에서 시작하게 되었으며, PDF를 업로드한 뒤 자연어로 질문하고 답변을 받을 수 있도록 설계했습니다.

---
😅 개발 후기

직접 해보니... 로컬에서 DeepSeek R1을 돌리는 건 꽤 느리고, 여기에 RAG까지 얹으니 정말 많이 느립니다.
하지만 좋은 개발 환경이 갖추어지면, 이 프로젝트를 더 발전시켜 실사용이 가능한 로컬 환경에서의 챗봇도 만들어 보고 싶습니다.

---

## 💡 주요 기능

- ✅ PDF 문서 업로드 및 자동 벡터화
- ✅ LangChain 기반 RAG 파이프라인 구성
- ✅ 로컬 LLM (`deepseek-r1`) 및 임베딩 모델 (`mxbai-embed-large`) 사용
- ✅ Streamlit UI 기반 실시간 질의응답

---

## 🖥️ 실행 방법

### 1. Ollama 설치
[Ollama를 설치](https://ollama.com/download)
하고 다음 명령어로 모델을 로컬에 다운로드
```
ollama pull deepseek-r1
ollama pull mxbai-embed-large
```

### 2. 환경 설치

```bash
pip install -r requirements.txt
```
### 3. 실행
```
streamlit run app.py
```
브라우저에서 http://localhost:8501 로 접속하여 사용

## 🔍 예시
| **문서에 관련 내용 있는 경우**                                                                 | **문서에 관련 내용 없는 경우**                                                         |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| ![image](https://github.com/user-attachments/assets/b406ee43-f76c-4ec3-8105-27bc9feabaf1)  | ![image](https://github.com/user-attachments/assets/39371469-cba2-4539-85ca-e49e6192b8b8)
 |


