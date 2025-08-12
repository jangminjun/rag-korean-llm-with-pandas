## 한국어지원 모델 테스트

### 1. 테스트 환경

- GPU Node :

- Jupyter Notebook Image : CUDA v12.1, Python v3.11

- LLM Model: ollama

- s3 Storeage : minio

- dataset : pandas dataset 활용

### 2. Jupyter Notebook Python Code

- 패키지 다운로드

  ```yaml
  # 필요한 라이브러리 설치
  !pip install pandas langchain langchain-community langchain-chroma sentence-transformers torch
  !pip install --upgrade pip
  !pip install langchain-huggingface
  !pip install pysqlite3-binary
  !pip install --upgrade chromadb langchain-community
  !pip install gradio
  ```
  
- 환경구성

  ```
  
  import os
  import sys
  import torch
  import boto3
  import pandas as pd
  import gradio as gr
  from langchain_community.document_loaders import DataFrameLoader
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_community.embeddings import HuggingFaceBgeEmbeddings
  from langchain_community.vectorstores import Chroma
  from langchain.prompts import PromptTemplate
  from langchain_community.chat_models import ChatOllama
  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.output_parser import StrOutputParser
  
  # pysqlite3 설정 (Chroma에서 sqlite3 이슈 방지)
  __import__('pysqlite3')
  sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
  
  ```

  

- 데이터셋 로드

  ```yaml
  # Titanic 데이터 불러오기
  csv_file_path = './data/titanic.csv'
  df = pd.read_csv(csv_file_path)
  
  # 데이터 가공
  df['Survived_str'] = df['Survived'].apply(lambda x: '생존' if x == 1 else '사망')
  df['combined_info'] = (
      df['Name'] + "은(는) " +
      df['Sex'] + "성 승객으로, 나이는 " + df['Age'].astype(str) + "세입니다. " +
      "탑승 등급은 " + df['Pclass'].astype(str) + "등급이었으며, 최종적으로 " +
      df['Survived_str'] + "했습니다."
  )
  
  # LangChain 문서 객체 변환
  loader = DataFrameLoader(df, page_content_column='combined_info')
  docs = loader.load()
  
  # 텍스트 분할
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
  split_docs = text_splitter.split_documents(docs)
  
  print(f"원본 문서 수: {len(docs)}")
  print(f"분할된 문서 수: {len(split_docs)}")
  
  ```

- 임베딩

  ```
  model_name = "BAAI/bge-m3"
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"임베딩 장치: {device}")
  
  embeddings = HuggingFaceBgeEmbeddings(
      model_name=model_name,
      model_kwargs={'device': device},
      encode_kwargs={'normalize_embeddings': True}
  )
  
  ```
  
  
  
- 모델 다운로드 및 minio 버킷에 업로드

  ```yaml
  # MinIO 환경 변수
  os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
  os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
  os.environ['S3_ENDPOINT_URL'] = 'https://minio-api-minio.apps.cluster-8882l.8882l.sandbox1647.opentlc.com'
  os.environ['S3_BUCKET'] = 'rag-with-langchain'
  
  # boto3 클라이언트 생성
  s3_client = boto3.client(
      's3',
      endpoint_url=os.environ['S3_ENDPOINT_URL'],
      aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
      aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
  )
  
  # 예시 모델 파일 업로드 (선택)
  local_file_path = 'my_model.pt'
  s3_file_key = 'models/my_model.pt'
  try:
      s3_client.upload_file(local_file_path, os.environ['S3_BUCKET'], s3_file_key)
      print("✅ 모델 파일 업로드 성공!")
  except FileNotFoundError:
      print(f"❌ '{local_file_path}' 파일이 없습니다.")
  
  ```
  
- 벡터스토어 s3 업로드

  ```
  vectorstore_dir = "./chroma_persist"
  
  try:
      vectorstore = Chroma.from_documents(
          documents=split_docs,
          embedding=embeddings,
          collection_name="rag-with-langchain",
          persist_directory=vectorstore_dir
      )
      vectorstore.persist()
  
      # 전체 파일 S3 업로드
      for root, dirs, files in os.walk(vectorstore_dir):
          for file in files:
              local_path = os.path.join(root, file)
              s3_path = os.path.relpath(local_path, vectorstore_dir)
              s3_client.upload_file(local_path, os.environ['S3_BUCKET'], s3_path)
  
      print(f"✅ MinIO 버킷 '{os.environ['S3_BUCKET']}'에 Chroma 데이터 저장 완료!")
  except Exception as e:
      print(f"❌ 저장 중 오류 발생: {e}")
  
  ```
  
  
  
- RAG 구성

  ```yaml
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  
  ollama_url = "http://ollama-server-service.ollama-dist.svc.cluster.local:11434"
  llm = ChatOllama(model="llama3", base_url=ollama_url)
  
  template = """
  당신은 사용자의 질문에 답하는 친절한 한국어 챗봇입니다.
  주어진 문맥(context)을 사용하여 질문(question)에 한국어로만 답하세요.
  만약 문맥에 관련 정보가 없거나 질문에 답할 수 없다면, 모른다고 답변하세요.
  
  문맥: {context}
  
  질문: {question}
  
  답변:
  """
  prompt = PromptTemplate.from_template(template)
  
  rag_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )
  
  ```
  
- 챗봇 UI 구성

  ```
  # ------------------------------------------------------------
  # 셀 5: Gradio UI 구성
  # ------------------------------------------------------------
  def chat_fn(message, history):
      try:
          answer = rag_chain.invoke(message)
          return answer
      except Exception as e:
          return f"⚠️ 오류 발생: {e}"
  
  with gr.Blocks() as demo:
      gr.Markdown("# 💬 RAG 기반 웹 챗봇")
      chatbot = gr.Chatbot()
      msg = gr.Textbox(label="질문을 입력하세요")
      clear = gr.Button("대화 초기화")
  
      def respond(user_message, chat_history):
          bot_message = chat_fn(user_message, chat_history)
          chat_history.append((user_message, bot_message))
          return "", chat_history
  
      msg.submit(respond, [msg, chatbot], [msg, chatbot])
      clear.click(lambda: None, None, chatbot, queue=False)
  
  ```
  
  

