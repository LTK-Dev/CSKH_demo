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
# <<< THAY ĐỔI: Trỏ đến file dữ liệu sản phẩm EKS
CSV_FILE = 'EKS.csv'
FAISS_INDEX_PATH = "faiss_index_eks" # <<< THAY ĐỔI: Thư mục lưu index riêng cho EKS
EMBEDDER_MODEL = 'intfloat/multilingual-e5-base'
# <<< THAY ĐỔI: Bạn có thể chọn model phù hợp, ví dụ 'gemini-1.5-flash'
GENERATIVE_MODEL = 'gemini-2.0-flash'

# --- 1. TỐI ƯU HIỆU NĂNG VỚI CACHING ---
@st.cache_resource
def get_embedder():
    """Tải và cache mô hình embedding."""
    print("INFO: Đang tải mô hình embedding...")
    return SentenceTransformerEmbeddings(
        model_name=EMBEDDER_MODEL,
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_or_create_faiss_index(_embedder):
    """
    Tải chỉ mục FAISS nếu đã tồn tại, nếu không thì tạo mới từ CSV
    sử dụng chiến thuật "Field Chunking".
    """
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"INFO: Đang tải chỉ mục FAISS từ '{FAISS_INDEX_PATH}'...")
        return FAISS.load_local(FAISS_INDEX_PATH, _embedder, allow_dangerous_deserialization=True)

    st.info("Chưa có chỉ mục FAISS. Bắt đầu quá trình tạo mới từ file CSV...")
    print("INFO: Bắt đầu tạo chỉ mục FAISS mới...")

    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        df.columns = [col.strip() for col in df.columns] # Làm sạch tên cột

        # <<< THAY ĐỔI LỚN: Áp dụng chiến thuật "Field Chunking"
        documents = []
        # Các cột chứa thông tin quan trọng cần chunk
        fields_to_chunk = [
            "CÔNG DỤNG", "THÀNH PHẦN", "CHỈ ĐỊNH",
            "CHỐNG CHỈ ĐỊNH", "CÁCH DÙNG", "BẢO QUẢN", "LƯU Ý KHI SỬ DỤNG"
        ]

        for _, row in df.iterrows():
            product_name = row.get("SẢN PHẨM", "Không rõ tên")
            for field in fields_to_chunk:
                # Chỉ tạo document nếu trường đó có dữ liệu
                if field in row and pd.notna(row[field]):
                    chunk_content = f"Sản phẩm: {product_name}\nThông tin về '{field}': {row[field]}"
                    documents.append(chunk_content)

        if not documents:
            st.error("Lỗi: Không thể tạo được các 'chunks' văn bản từ file CSV. Vui lòng kiểm tra lại cấu trúc file và tên các cột.")
            st.stop()
        
        print(f"INFO: Đã tạo được {len(documents)} chunks từ file CSV.")
        print("INFO: Đang tạo embeddings và chỉ mục FAISS...")
        vectorstore = FAISS.from_texts(texts=documents, embedding=_embedder)
        
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"INFO: Tạo và lưu chỉ mục FAISS vào '{FAISS_INDEX_PATH}' thành công.")
        st.success("Tạo chỉ mục sản phẩm thành công! Chatbot đã sẵn sàng.")
        # Không cần rerun vì cache resource sẽ xử lý việc load lại
        return vectorstore

    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{CSV_FILE}'. Vui lòng đảm bảo file tồn tại.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tạo chỉ mục FAISS: {e}")
        st.stop()

# --- 2. CẤU HÌNH API KEY AN TOÀN ---
try:
    # Lấy API key từ Streamlit Secrets khi deploy
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Lỗi: Vui lòng thiết lập `GEMINI_API_KEY` trong phần Secrets của Streamlit.")
    st.stop()

# --- Tải tài nguyên và khởi tạo ---
embedder = get_embedder()
vector_store = load_or_create_faiss_index(embedder)
model = genai.GenerativeModel(GENERATIVE_MODEL)

# --- 3. LOGIC XỬ LÝ CHÍNH ---
def get_relevant_documents(question: str, n_results: int = 5) -> List[str]: # <<< THAY ĐỔI: Tăng số lượng context
    """Sử dụng FAISS để truy vấn các chunks tài liệu liên quan."""
    results = vector_store.similarity_search(question, k=n_results)
    return [doc.page_content for doc in results]

def generate_response_stream(question: str, context: List[str]):
    # <<< THAY ĐỔI: Cập nhật prompt để phù hợp với EKS và field chunking
    prompt_template = ChatPromptTemplate.from_template(
        """
        Bạn là trợ lý ảo chuyên nghiệp của nhãn hàng mỹ phẩm Ekseption (EKS). Nhiệm vụ của bạn là tư vấn chính xác và thân thiện cho khách hàng dựa trên thông tin sản phẩm được cung cấp.

        **Thông tin tham khảo từ các sản phẩm (đã được chia nhỏ theo từng trường thông tin)**:
        ---
        {context}
        ---

        **Câu hỏi của khách hàng**:
        {question}

        ---
        **Hướng dẫn trả lời**:
        1.  **Xưng hô**: Luôn xưng là "EKS" và gọi khách hàng là "bạn". Giọng điệu chuyên gia, tự tin nhưng gần gũi.
        2.  **Tổng hợp thông tin**: Dựa vào các mảnh thông tin trong phần "Thông tin tham khảo" để tổng hợp câu trả lời đầy đủ nhất. Các mảnh thông tin này có thể đến từ nhiều sản phẩm khác nhau, hãy chọn lọc thông tin đúng với sản phẩm mà khách hàng hỏi.
        3.  **Trích dẫn sản phẩm**: Khi trả lời, hãy nêu rõ thông tin đó thuộc về sản phẩm nào. Ví dụ: "Đối với sản phẩm [Tên sản phẩm], công dụng của nó là..."
        4.  **Nếu không chắc chắn**: Nếu thông tin tham khảo không đủ để trả lời câu hỏi, hãy trả lời rằng: "Cảm ơn câu hỏi của bạn. EKS chưa có thông tin chi tiết về vấn đề này trong dữ liệu. Bạn vui lòng cung cấp thêm chi tiết hoặc liên hệ chuyên gia để được tư vấn sâu hơn nhé." Tuyệt đối không tự bịa đặt thông tin.
        5.  **Ngôn ngữ**: Sử dụng tiếng Việt, trình bày rõ ràng, dùng markdown (gạch đầu dòng, in đậm) để câu trả lời dễ đọc.
        """
    )
    
    formatted_prompt = prompt_template.format(context="\n\n".join(context), question=question)
    response_stream = model.generate_content(formatted_prompt, stream=True)
    for chunk in response_stream:
        yield chunk.text

# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
# <<< THAY ĐỔI: Cập nhật giao diện cho EKS
st.set_page_config(page_title="EKS Chatbot", page_icon="✨")
st.title("💬 Chatbot Tư vấn sản phẩm EKS")
st.caption("Trợ lý ảo thông minh của Ekseption (Nền tảng: Field Chunking + FAISS + Gemini)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, EKS có thể giúp gì cho bạn hôm nay? Hãy hỏi tôi bất cứ điều gì về sản phẩm nhé!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hỏi về công dụng, thành phần, cách dùng..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("EKS đang tìm kiếm câu trả lời..."):
            context = get_relevant_documents(prompt)
            
            # (Tùy chọn) In context ra để debug
            # with st.expander("Xem context được tìm thấy"):
            #     st.write(context)

            response_generator = generate_response_stream(prompt, context)
            full_response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
