# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- KHỞI TẠO STATE CHO KHUNG CHAT ---
# Khởi tạo lịch sử chat nếu chưa tồn tại
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# Khởi tạo dữ liệu phân tích để gửi cho AI nếu chưa tồn tại
if "data_for_ai_markdown" not in st.session_state:
    st.session_state["data_for_ai_markdown"] = ""
# Khởi tạo client AI
if "ai_client" not in st.session_state:
    st.session_state["ai_client"] = None


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét ban đầu."""
    try:
        # Khởi tạo client và lưu vào session_state
        client = genai.Client(api_key=api_key)
        st.session_state["ai_client"] = client
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Hàm xử lý khung chat ---
def handle_chat_input(prompt_user):
    """Xử lý đầu vào từ người dùng, gọi API và cập nhật lịch sử chat."""
    
    # Thêm tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    
    # Tạo bối cảnh (context) cho AI, bao gồm dữ liệu phân tích và lịch sử chat
    chat_context = [
        # Vai trò hệ thống
        {"role": "user", "parts": [
            f"Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Hãy trả lời các câu hỏi dựa trên dữ liệu phân tích sau: {st.session_state.data_for_ai_markdown}"
        ]},
        {"role": "model", "parts": ["Đã hiểu, tôi sẽ sử dụng dữ liệu này để trả lời các câu hỏi tiếp theo."]}
    ]
    
    # Thêm các tin nhắn cũ vào bối cảnh (trừ tin nhắn hệ thống trên)
    for message in st.session_state.messages:
        chat_context.append(message)
        
    # Lấy client đã được lưu
    client = st.session_state.ai_client

    if client:
        try:
            with st.spinner("Đang suy nghĩ..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=chat_context
                )
            
            ai_response = response.text
            # Thêm tin nhắn của AI vào lịch sử
            st.session_state.messages.append({"role": "model", "content": ai_response})
            return ai_response
            
        except Exception as e:
            error_msg = f"Lỗi trong quá trình trò chuyện: {e}"
            st.session_state.messages.append({"role": "model", "content": error_msg})
            return error_msg
    else:
        # Trường hợp hiếm gặp nếu client chưa được khởi tạo
        error_msg = "Lỗi: Client AI chưa được khởi tạo. Vui lòng kiểm tra lại cấu hình API."
        st.session_state.messages.append({"role": "model", "content": error_msg})
        return error_msg

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        # RESET state khi file mới được tải lên
        st.session_state["messages"] = [] 
        st.session_state["data_for_ai_markdown"] = ""
        st.session_state["ai_client"] = None
        
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # ... (Giữ nguyên Chức năng 2, 3, 4) ...
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Tính Chỉ số Thanh toán Hiện hành (Tương tự code gốc)
            try:
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Chỉ số Thanh toán Hiện hành (Năm trước)", value=f"{thanh_toan_hien_hanh_N_1:.2f} lần")
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                st.error("Lỗi: Nợ Ngắn Hạn bằng 0, không thể tính chỉ số Thanh toán Hiện hành.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chuẩn bị dữ liệu cho AI ---
            data_for_ai_df = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if any(df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)) else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            })
            # LƯU DỮ LIỆU ĐỂ DÙNG TRONG KHUNG CHAT
            st.session_state["data_for_ai_markdown"] = data_for_ai_df.to_markdown(index=False)
            
            
            # --- Chức năng 5: Nhận xét AI (giữ nguyên) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(st.session_state["data_for_ai_markdown"], api_key)
                        
                        # Khởi tạo tin nhắn chat đầu tiên
                        st.session_state.messages.append({"role": "model", "content": ai_result})
                        
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            
            # --- CHỨC NĂNG 6: KHUNG CHAT THẢO LUẬN VỚI AI ---
            if st.session_state.get("ai_client"):
                st.subheader("6. Hỏi & Đáp về Báo cáo Tài chính (Chat)")
                
                # 1. Hiển thị lịch sử chat
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # 2. Xử lý đầu vào người dùng
                # Chỉ hiển thị khung nhập nếu có client AI (đã phân tích bước 5)
                if prompt := st.chat_input("Hỏi thêm về Báo cáo Tài chính này..."):
                    handle_chat_input(prompt)
                    # Tự động chạy lại ứng dụng để hiển thị tin nhắn mới
                    st.rerun()

            
    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
