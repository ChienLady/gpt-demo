import streamlit as st
import openai
import re
import os

finetune_model = {
    'temperature': 0.0,
    'top_p': 0.0,
    'max_tokens': 1000,
    'model': 'davinci:ft-personal-2023-02-13-12-02-57',
    'stream': True
}
rep_model = finetune_model.copy()

def param_input():
    rep_model['temperature'] = st.sidebar.slider(
        'temperature', min_value = 0.0, max_value = 2.0, value = finetune_model['temperature'], step = 0.01
    )
    rep_model['top_p'] = st.sidebar.slider(
        'top_p', min_value = 0.0, max_value = 1.0, value = finetune_model['top_p'], step = 0.01
    )

def main():
    openai.api_key = os.environ.get('OPENAI_KEY2')
    param_input()
    st.markdown(
        '''
<h1 align="center">
    🔥 Fine-tuned model
</h1>
        ''',
        unsafe_allow_html = True,
    )
    
    default_value = 'Cách khai báo loại tiền mới trên web?'
    question = st.text_input('Câu hỏi:', default_value)
    st.write('Trả lời:')
    answer = st.empty()
    answer.markdown('')
    
    prompt = f'Câu hỏi: {question}\nTrả lời: '
    if st.button('Lấy câu trả lời'):
        tokens = 0
        with st.spinner('Đang sinh câu trả lời...'):
            response = ''
            for resp in openai.Completion.create(prompt = prompt, **rep_model):
                tokens += 1
                response += resp.choices[0].text
                response = response.replace(r'\n', '\n\n')
                links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response)
                if len(links) > 0:
                    for link in links:
                        link = link.replace('[', '').replace(']', '')
                        if link[-4:] in ['.jpg', '.png', 'jpeg', '.gif']:
                            response = response.replace(link, f'![image]({link})')
                        if link.startswith('https://vimeo.com'):
                            id = link[link.rindex('/') + 1:]
                            response = response.replace(link, f'<iframe src="https://player.vimeo.com/video/{id}?autoplay=1&loop=1&title=0&byline=0&portrait=0" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>')
                try:
                    answer.markdown(response)
                except:
                    pass
        st.success(f'Đã tạo xong câu trả lời gồm {tokens} tokens tiêu tốn {0.12 * tokens / 1000}$')
