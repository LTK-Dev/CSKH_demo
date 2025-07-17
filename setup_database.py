# setup_database.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st # Thêm streamlit để hiển thị thông báo

# --- Cấu hình ---
CSV_FILE = 'hahaha_output.csv'
DB_PATH = "./chroma_db"
COLLECTION_NAME = "docilee_data"
EMBEDDER_MODEL = 'BAAI/bge-m3'

def initialize_database():
    """
    Hàm này được thiết kế để app.py gọi khi cần.
    Nó sẽ tạo cơ sở dữ liệu vector từ file CSV.
    """
    st.info(f"Cơ sở dữ liệu tại '{DB_PATH}' chưa có. Bắt đầu quá trình khởi tạo...")
    print(f"INFO: Bắt đầu quá trình khởi tạo DB tại '{DB_PATH}'...")

    try:
        # Tải mô hình embedder
        print(f"INFO: Đang tải mô hình embedder: {EMBEDDER_MODEL}...")
        # Dùng 'cpu' để đảm bảo tương thích trên môi trường deploy
        embedder = SentenceTransformer(EMBEDDER_MODEL, device='cpu') 

        # Đọc và xử lý file CSV
        print(f"INFO: Đang đọc và xử lý file: {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE, encoding='utf-8')

        documents = [
            f"Câu hỏi: {row['Câu hỏi']}\nTrả lời: {row['Trả lời']}"
            for _, row in df.iterrows()
            if pd.notna(row['Câu hỏi']) or pd.notna(row['Trả lời'])
        ]
        ids = [str(i) for i in range(len(documents))]

        if not documents:
            st.error("Lỗi: Không tìm thấy dữ liệu hợp lệ trong file CSV.")
            st.stop()
            return # Dừng hàm

        # Tạo embeddings
        print("INFO: Đang tạo embeddings... (Quá trình này có thể mất vài phút)")
        embeddings = embedder.encode(documents, normalize_embeddings=True, show_progress_bar=True)

        # Khởi tạo và lưu vào ChromaDB
        print(f"INFO: Đang lưu dữ liệu vào ChromaDB tại: {DB_PATH}...")
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        
        collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=ids
        )

        print(f" INFO: Khởi tạo cơ sở dữ liệu thành công với {collection.count()} tài liệu.")
        st.success("Khởi tạo cơ sở dữ liệu thành công!")

    except FileNotFoundError:
        st.error(f"Lỗi deploy: Không tìm thấy file '{CSV_FILE}'. Hãy chắc chắn bạn đã tải file này lên GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi khởi tạo cơ sở dữ liệu: {e}")
        st.stop()