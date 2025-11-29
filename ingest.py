import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 로드
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-rules")
NAMESPACE = "rules-2025"

def parse_rules(file_path):
    """
    텍스트 파일을 읽어 '제N조' 단위로 문서를 분할합니다.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 정규표현식: "제1조", "제 2 조" 등 조항 시작 패턴 감지
    # 패턴 설명: 줄바꿈 뒤에 '제', 숫자, '조' 가 오는 경우를 기준으로 자름
    pattern = r'(\n|^)제\s?\d+\s?조'
    
    # split으로 나누면 [내용, 조항제목, 내용, 조항제목...] 순서로 나옴
    # 좀 더 쉬운 처리를 위해 '제N조' 위치를 찾아 수동으로 슬라이싱합니다.
    matches = list(re.finditer(pattern, full_text))
    
    documents = []
    for i, match in enumerate(matches):
        start = match.start()
        # 다음 조항 시작 전까지가 현재 조항의 내용
        end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        
        content = full_text[start:end].strip()
        
        # 첫 번째 줄(예: 제1조(목적))을 추출하여 메타데이터로 활용
        lines = content.split('\n')
        title = lines[0].strip() if lines else "Unknown"
        
        # 문서 객체 생성
        doc = Document(
            page_content=content,
            metadata={
                "source": "취업규칙(2025)",
                "article_title": title,
                "category": "규정" # 필요시 카테고리 로직 추가 가능
            }
        )
        documents.append(doc)
        
    return documents

def ingest_data():
    print(f"🚀 데이터 파싱 시작... (Namespace: {NAMESPACE})")
    
    # 1. 데이터 파싱
    file_path = "rules.txt"
    if not os.path.exists(file_path):
        print("❌ rules.txt 파일이 없습니다.")
        return

    docs = parse_rules(file_path)
    print(f"✅ 총 {len(docs)}개의 조항(Chunk)으로 분할되었습니다.")
    print(f"   - 예시: {docs[0].page_content[:50]}...")

    # 2. 임베딩 모델 준비
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3. Pinecone에 업로드 (LangChain Wrapper 사용)
    # 기존 데이터 충돌 방지를 위해, 해당 네임스페이스를 비우는 로직은 Pinecone 클라이언트로 직접 처리하거나
    # 덮어쓰기 로직을 고민해야 합니다. 여기서는 Upsert 방식으로 진행합니다.
    
    print("📡 Pinecone 업로드 중...")
    
    vector_store = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE
    )
    
    print("🎉 업로드 완료!")

if __name__ == "__main__":
    ingest_data()