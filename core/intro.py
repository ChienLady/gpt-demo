import streamlit as st

def main():
    st.markdown(
        '''
<h1 align="center">
    Question-Answering Demo site 👋
</h1>
        ''',
        unsafe_allow_html = True,
    )

    with st.expander('Giới thiệu', True):
        st.markdown(
            '''
Demo sử dụng phương pháp **Embeddings** để tìm bài viết liên quan nhất đến câu hỏi tạo thành ngữ cảnh sau đó sử dụng API của **InstructGPT** để tạo câu trả lời.

<img src="https://lh3.googleusercontent.com/7pDo9u-QNevF0FY-4NzgEk_IgTJipLwVItlrpAnEVzIaEpgiA7F-YOnglbgSuA20zUxnXdVAwXTss0VRilHnSCmc2OMsC7cnhYmRxNyOsrBmeLU057jiiSTcDlAgJoKqmD8KJ5cNZNsqSrI9Tyfr35UeApoZ8z687LviNJsSBHkAUqkpaP5XrgMKL43YnQ" alt="InstructGPT" width="80%" height="80%" style="display: block;margin-left: auto;margin-right: auto">

---
Hiện tại demo đang trong quá trình hoàn thiện các bài ngữ cảnh, hiện đã hỗ trợ cho một số sản phẩm sau:
> 1. [MISA eShop](https://eshop.misa.vn/)
> 2. [MISA eSign](https://esign.misa.vn/)
> 3. [MISA Lending](https://lending.misa.vn/)
> 4. [MISA AMIS Kế toán](https://amis.misa.vn/amis-ke-toan/)
> 5. [MISA mTax](https://mtax.misa.vn/)
> 6. [MISA AMIS Tuyển dụng](https://amis.misa.vn/amis-tuyen-dung/)
> 7. [MISA BankHub](https://bankhub.misa.vn/)
> 8. [MISA SME](https://sme.misa.vn/)
> 9. [MISA AMIS aiMarketing](https://amis.misa.vn/amis-aimarketing/)
> 10. [MISA AMIS Khuyến mại](https://amis.misa.vn/amis-khuyen-mai/)
> 11. [MISA ASP](https://asp.misa.vn/)
> 12. [MISA AMIS CRM](https://amis.misa.vn/phan-mem-crm-amis/)
> 13. [MISA AMIS Thông tin nhân sự](https://amis.misa.vn/thong-tin-nhan-su/)
> 14. ...

<img src="http://hanoimoi.com.vn/Uploads/images/quangcao2/2022/11/30/384BBF25-E0EE-4749-8302-8BCD951D9FA8.png" alt="InstructGPT" width="80%" height="80%" style="display: block;margin-left: auto;margin-right: auto">

---

<p align="middle"> Writen: <b> AITeam </b> </p>
<p align="middle"> Copyright © 1994 - 2023 MISA JSC </p>              
            ''',
            unsafe_allow_html = True
        )
    
if __name__ == '__main__':
    main()

