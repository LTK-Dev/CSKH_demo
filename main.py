# # app.py
# import streamlit as st
# import chromadb
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from langchain_core.prompts import ChatPromptTemplate
# from typing import List
# import os
# from chromadb.config import Settings

# # --- 1. IMPORT VÀ GỌI HÀM SETUP ---
# from setup_database import initialize_database, DB_PATH # Import hàm và biến đường dẫn

# # Kiểm tra và khởi tạo DB nếu cần
# if not os.path.exists(DB_PATH):
#     initialize_database()
#     st.rerun() # Tải lại ứng dụng sau khi DB được tạo để đảm bảo mọi thứ được load đúng


# # --- 2. CẤU HÌNH API KEY AN TOÀN ---
# try:
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
#     genai.configure(api_key=GEMINI_API_KEY)
# except KeyError:
#     st.error("Lỗi: Vui lòng thiết lập `GEMINI_API_KEY` trong phần Secrets của Streamlit.")
#     st.stop()


# # --- PHẦN CÒN LẠI CỦA APP.PY GIỮ NGUYÊN ---

# # --- 3. TỐI ƯU HIỆU NĂNG VỚI CACHING ---
# COLLECTION_NAME = "docilee_data"
# EMBEDDER_MODEL = 'BAAI/bge-m3'
# GENERATIVE_MODEL = 'gemini-2.0-flash'

# @st.cache_resource
# def get_embedder():
#     print("INFO: Đang tải mô hình embedding...")
#     return SentenceTransformer(EMBEDDER_MODEL)

# @st.cache_resource
# def get_retriever():
#     print("INFO: Đang kết nối tới ChromaDB...")
#     # Sửa dòng client = chromadb.PersistentClient(path=DB_PATH) thành:
#     client = chromadb.PersistentClient(
#         path=DB_PATH,
#         settings=Settings(
#             allow_reset=True,
#             anonymized_telemetry=False
#         )
#     )
#     return client.get_collection(name=COLLECTION_NAME)

# @st.cache_resource
# def get_generative_model():
#     print("INFO: Đang khởi tạo mô hình Gemini...")
#     return genai.GenerativeModel(GENERATIVE_MODEL)

# embedder = get_embedder()
# retriever = get_retriever()
# model = get_generative_model()

# # --- 4. LOGIC XỬ LÝ CHÍNH ---
# def get_relevant_documents(question: str, n_results: int = 3) -> List[str]:
#     query_embedding = embedder.encode([question], normalize_embeddings=True)
#     results = retriever.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
#     return results['documents'][0] if results.get('documents') else []

# def generate_response_stream(question: str, context: List[str]):
#     prompt_template = ChatPromptTemplate.from_template(
#         """
#         Bạn là trợ lý ảo của thương hiệu Docilee, chuyên về sản phẩm cho mẹ và bé. Trả lời câu hỏi của khách hàng một cách thân thiện, chuyên nghiệp và chính xác dựa trên thông tin sau.

#         **Thông tin tham khảo**:
#         {context}

#         ---
#         **Câu hỏi của khách hàng**:
#         {question}

#         ---
#         **Hướng dẫn trả lời**:
#         - Luôn trả lời bằng tiếng Việt, giọng điệu gần gũi như một chuyên gia tư vấn.
#         - Quan trọng: Nếu thông tin tham khảo có câu hỏi nào tương ứng với câu hỏi của khách hàng, hãy trả lời y hệt phần Trả lời của câu hỏi đó.
#         - Nếu câu hỏi trong phần Thông tin tham khảo không hoàn toàn giống về mặt ngữ nghĩa với khách hàng thì hãy trả lời theo đúng kiến thức mà bạn có.
#         - Nếu thông tin không đủ hoặc không trả lời được chính xác câu hỏi, hãy trả lời một cách tổng quát dựa trên kiến thức chung về sản phẩm Docilee (tã, bỉm, an toàn cho da bé,...) và thừa nhận rằng bạn không có thông tin chi tiết.
#         - Giữ câu trả lời súc tích, đi thẳng vào vấn đề.

#         Nếu không tuân thủ đúng hướng dẫn, bạn sẽ bị phạt.
#         """
#     )
    
#     formatted_prompt = prompt_template.format(context="\n\n".join(context), question=question)
#     print("--- PROMPT ĐÃ GỬI TỚI GEMINI ---")
#     print(formatted_prompt)
#     print("---------------------------------")
    
#     response_stream = model.generate_content(formatted_prompt, stream=True)
#     for chunk in response_stream:
#         yield chunk.text

# # --- 5. GIAO DIỆN NGƯỜI DÙNG ---
# st.set_page_config(page_title="Docilee Chatbot", page_icon="👶")
# st.title("💬 Docilee AI Chatbot")
# st.caption("Trợ lý ảo thông minh của Docilee luôn sẵn sàng hỗ trợ bạn!")

# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, mình là trợ lý ảo của Docilee. Bạn cần tư vấn về sản phẩm nào ạ?"}]

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if prompt := st.chat_input("Nhập câu hỏi của bạn về sản phẩm Docilee..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Docilee đang suy nghĩ..."):
#             context = get_relevant_documents(prompt)
#             response_generator = generate_response_stream(prompt, context)
#             full_response = st.write_stream(response_generator)

#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# app.py (Phiên bản cuối cùng, hoàn chỉnh sử dụng FAISS)
import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from typing import List

# Import các thư viện cần thiết cho FAISS và LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- Cấu hình ---
CSV_FILE = 'hahaha_output.csv'
FAISS_INDEX_PATH = "faiss_index" # Thư mục để lưu chỉ mục FAISS
EMBEDDER_MODEL = 'intfloat/multilingual-e5-base'
GENERATIVE_MODEL = 'gemini-2.0-flash'

# --- 1. TỐI ƯU HIỆU NĂNG VỚI CACHING ---
@st.cache_resource
def get_embedder():
    """Tải và cache mô hình embedding."""
    print("INFO: Đang tải mô hình embedding...")
    # Dùng wrapper của LangChain để tương thích tốt hơn
    # 'cpu' để đảm bảo tương thích trên môi trường deploy
    return SentenceTransformerEmbeddings(
        model_name=EMBEDDER_MODEL,
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_or_create_faiss_index(_embedder):
    """
    Tải chỉ mục FAISS nếu đã tồn tại, nếu không thì tạo mới từ CSV.
    Hàm này sẽ chạy một lần khi app khởi động.
    """
    # 'allow_dangerous_deserialization=True' là cần thiết cho LangChain phiên bản mới
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"INFO: Đang tải chỉ mục FAISS từ '{FAISS_INDEX_PATH}'...")
        return FAISS.load_local(FAISS_INDEX_PATH, _embedder, allow_dangerous_deserialization=True)

    st.info("Chưa có chỉ mục FAISS. Bắt đầu quá trình tạo mới từ file CSV...")
    print("INFO: Bắt đầu tạo chỉ mục FAISS mới...")

    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        # Tạo nội dung cho các documents
        documents = [
            f"Câu hỏi: {row['Câu hỏi']}\nTrả lời: {row['Trả lời']}"
            for _, row in df.iterrows() if pd.notna(row['Câu hỏi']) or pd.notna(row['Trả lời'])
        ]
        if not documents:
            st.error("Lỗi: Không tìm thấy dữ liệu hợp lệ trong file CSV.")
            st.stop()
        
        # Tạo chỉ mục vector từ các document và mô hình embedding
        print("INFO: Đang tạo embeddings và chỉ mục FAISS...")
        vectorstore = FAISS.from_texts(texts=documents, embedding=_embedder)
        
        # Lưu chỉ mục ra file để lần sau tải lại cho nhanh
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(" INFO: Tạo và lưu chỉ mục FAISS thành công.")
        st.success("Tạo chỉ mục thành công! Chatbot đã sẵn sàng.")
        st.rerun() # Tải lại app sau khi tạo xong
        return vectorstore

    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{CSV_FILE}'. Vui lòng tải file này lên GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tạo chỉ mục FAISS: {e}")
        st.stop()


# --- 2. CẤU HÌNH API KEY AN TOÀN ---
try:
    # Lấy API key từ Streamlit Secrets khi deploy
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Lỗi: Vui lòng thiết lập `GEMINI_API_KEY` trong phần Secrets của Streamlit.")
    st.stop()


# --- Tải tài nguyên và khởi tạo ---
embedder = get_embedder()
vector_store = load_or_create_faiss_index(embedder)
model = genai.GenerativeModel(GENERATIVE_MODEL)

# --- 3. LOGIC XỬ LÝ CHÍNH ---
def get_relevant_documents(question: str, n_results: int = 3) -> List[str]:
    """Sử dụng FAISS để truy vấn các tài liệu liên quan."""
    # similarity_search trả về các đối tượng Document của LangChain
    results = vector_store.similarity_search(question, k=n_results)
    # Trích xuất nội dung text từ các document
    return [doc.page_content for doc in results]

def generate_response_stream(question: str, context: List[str]):
    prompt_template = ChatPromptTemplate.from_template(
        """
        Bạn là trợ lý ảo của thương hiệu Docilee, chuyên về sản phẩm cho mẹ và bé. Trả lời câu hỏi của khách hàng một cách thân thiện, chuyên nghiệp và chính xác dựa trên thông tin sau.

        **Thông tin tham khảo**:
        {context}

        ---
        **Câu hỏi của khách hàng**:
        {question}

        ---
        **Hướng dẫn trả lời**:
        - Luôn trả lời bằng tiếng Việt, giọng điệu gần gũi như một chuyên gia tư vấn.
        - Quan trọng: Nếu thông tin tham khảo có câu hỏi nào tương ứng với câu hỏi của khách hàng, hãy trả lời y hệt phần Trả lời của câu hỏi đó.
        - Nếu câu hỏi trong phần Thông tin tham khảo không hoàn toàn giống về mặt ngữ nghĩa với khách hàng thì hãy trả lời theo đúng kiến thức mà bạn có.
        - Nếu thông tin không đủ hoặc không trả lời được chính xác câu hỏi, hãy trả lời một cách tổng quát dựa trên kiến thức chung về sản phẩm Docilee (tã, bỉm, an toàn cho da bé,...) và thừa nhận rằng bạn không có thông tin chi tiết.
        - Giữ câu trả lời súc tích, đi thẳng vào vấn đề.

        Nếu không tuân thủ đúng hướng dẫn, bạn sẽ bị phạt.
        """
    )
    
    formatted_prompt = prompt_template.format(context="\n\n".join(context), question=question)
    response_stream = model.generate_content(formatted_prompt, stream=True)
    for chunk in response_stream:
        yield chunk.text


# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
st.set_page_config(page_title="Docilee Chatbot", page_icon="👶")
st.title("💬 Docilee AI Chatbot")
st.caption("Trợ lý ảo thông minh của Docilee (Nền tảng: FAISS + Gemini)")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, mình là trợ lý ảo của Docilee. Bạn cần tư vấn về sản phẩm nào ạ?"}]

# Hiển thị các tin nhắn đã có
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Khung nhập liệu cho người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về sản phẩm Docilee..."):
    # Thêm và hiển thị tin nhắn của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tạo và hiển thị câu trả lời của bot
    with st.chat_message("assistant"):
        with st.spinner("Docilee đang suy nghĩ..."):
            # 1. Tìm kiếm thông tin liên quan
            context = get_relevant_documents(prompt)
            
            # 2. Tạo và hiển thị câu trả lời theo từng phần (streaming)
            response_generator = generate_response_stream(prompt, context)
            full_response = st.write_stream(response_generator)

    # Thêm câu trả lời hoàn chỉnh của bot vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})