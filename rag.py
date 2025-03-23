# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader  # 이 부분에서 오류 발생 시 PDF 관련 의존성 확인 필요
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

# 디버깅 및 로깅 설정
set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """PDF를 처리하고 RAG 방식으로 질문에 답변하는 클래스"""

    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        """
        LLM과 임베딩 모델을 초기화합니다.
        """
        self.model = ChatOllama(model=llm_model)  # 언어 모델 초기화
        self.embeddings = OllamaEmbeddings(model=embedding_model)  # 임베딩 모델 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)  # 문서 분할 설정

        # LLM 프롬프트 템플릿 정의
        self.prompt = ChatPromptTemplate.from_template(
            """
            당신은 업로드된 문서를 바탕으로 질문에 답하는 유용한 도우미입니다. 반드시 한국어로 답해주세요.
            문서에 관련 내용이 없으면 '모르겠습니다.'라고 답하세요.

            문맥:
            {context}
            
            질문:
            {question}
            
            세 문장 이내로 간결하고 정확하게 답변해주세요.
            """
        )

        self.vector_store = None  # 벡터 저장소 초기화
        self.retriever = None     # 검색기 초기화

    def ingest(self, pdf_file_path: str) -> dict:
        """
        PDF 파일을 벡터로 변환하여 저장소에 저장합니다.
        """
        try:
            logger.info(f"[ingest 시작] 파일 경로: {pdf_file_path}")
            docs = PyPDFLoader(file_path=pdf_file_path).load()  # PDF 문서 로딩
            chunks = self.text_splitter.split_documents(docs)  # 문서 분할
            chunks = filter_complex_metadata(chunks)  # 복잡한 메타데이터 필터링
            
            logger.info(f"총 {len(chunks)}개의 청크 생성됨")
            logger.info(f"샘플 청크 내용: {chunks[0].page_content[:100]}...")

            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
            )

            # 벡터 저장소 문서 수 확인
            collection_size = self.vector_store._collection.count()
            logger.info(f"벡터 저장소 문서 수: {collection_size}")

            logger.info("ingest 완료")
            return {"status": "success", "message": f"성공적으로 {len(chunks)}개의 문서를 인제스트했습니다."}
        except Exception as e:
            logger.error(f"[ingest 오류] {str(e)}")
            return {"status": "error", "message": str(e)}

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        사용자 질문에 대해 RAG 파이프라인을 사용해 답변을 생성합니다.
        """
        if not self.vector_store:
            raise ValueError("벡터 저장소가 없습니다. 먼저 문서를 인제스트하세요.")

        # 검색기 초기화
        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"질문에 대한 문맥 검색: {query}")
        retrieved_docs = self.retriever.invoke(query)  # 관련 문서 검색

        if not retrieved_docs:
            return "문서에서 해당 질문에 대한 관련 문맥을 찾을 수 없습니다."

        # 🔍 검색된 문서 원문 로그 출력
        logger.info(f"총 {len(retrieved_docs)}개의 문서가 검색되었습니다.")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"[문서 {i+1}] {doc.page_content[:500]}...")  # 500자까지만 출력 (필요 시 조절)

        # LLM 입력 포맷 생성
        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # RAG 체인 구성
        chain = (
            RunnablePassthrough()
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        logger.info("LLM을 사용하여 응답 생성 중.")
        return chain.invoke(formatted_input)


    def clear(self):
        """
        벡터 저장소와 검색기를 초기화합니다.
        """
        logger.info("벡터 저장소 및 검색기 초기화됨")
        self.vector_store = None
        self.retriever = None
