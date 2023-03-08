import streamlit as st
from .process import get_document, COMPLETIONS_API_PARAMS, count_tokens
import openai
import re
import os
import pandas as pd
    
REPLACE_API_PARAMS = COMPLETIONS_API_PARAMS.copy()

def key_input():
    rkey = st.sidebar.text_input('OpenAI Key')
    if st.sidebar.checkbox('Key thay thế'):
        openai.api_key = rkey

def param_input():
    REPLACE_API_PARAMS['temperature'] = st.sidebar.slider(
        'temperature', min_value = 0.0, max_value = 2.0, value = COMPLETIONS_API_PARAMS['temperature'], step = 0.01
    )
    REPLACE_API_PARAMS['top_p'] = st.sidebar.slider(
        'top_p', min_value = 0.0, max_value = 1.0, value = COMPLETIONS_API_PARAMS['top_p'], step = 0.01
    )

def parse_response(response, used):
    response = response.replace(r'\n', '\n\n')
    links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response)
    if len(links) > 0:
        for link in links:
            if link[-4:] in ['.jpg', '.png', 'jpeg', '.gif'] and link not in used:
                used.append(link)
                response = response.replace(link, f'![image]({link})')
            elif link.startswith('https://player.vimeo.com/video/') and link not in used and len(link.rsplit('/', 1)[-1]) == 9:
                link = link.strip('.')
                used.append(link)
                response = response.replace(link, f'<iframe src="{link}?autoplay=1&loop=1&title=0&byline=0&portrait=0" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>')
            elif link.startswith('https://www.youtube.com/embed/') and link not in used and len(link.rsplit('/', 1)[-1]) == 11:
                link = link.strip('.')
                used.append(link)
                response = response.replace(link, f'<iframe src="{link}" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>')
    return response

def df_history(data: list):
    roles, contents = [{k: [d[k].replace('\n', r'\n') for d in data]} for k in ['role', 'content']]
    roles.update(contents)
    
    return pd.DataFrame(roles, index = range(1, len(data) + 1))

def main():
    key = os.environ.get('OPENAI_KEY')
    if key is not None:
        openai.api_key = key
        st.success('**Key** hiện có thể sử dụng, không cần nhập **Key** thay thế!')
    else:
        
        st.error('Không có **Key**, vui lòng nhập **Key** thay thế!')
    key_input()
    param_input()
    st.markdown(
        '''
<h1 align="center">
    🔥 Hệ thống hỏi đáp
</h1>
        ''',
        unsafe_allow_html = True,
    )
    if 'message' not in st.session_state:
        st.session_state['message'] = [
            {'role': 'system', 'content': 'Hướng dẫn: Trả lời chi tiết dựa vào tri thức (chỉ đưa ra link http và ký tự "\\n" nếu có trong tri thức của MISA)\nChú ý: Nếu câu trả lời không ở trong tri thức MISA, tự trả lời theo tri thức của mình.'}
        ]
    
    default_value = 'Các gói sản phẩm SME?'
    question = st.text_input('Câu hỏi:', default_value)
    with st.expander('Messages', False):
        message_container = st.empty()
        message_container.dataframe(df_history(st.session_state.message), use_container_width = True)
    with st.expander('Context', False):
        context = st.empty()
        context.markdown('')
    if st.button('Reset context'):
        st.session_state['message'] = [
            {'role': 'system', 'content': 'Hướng dẫn: Trả lời chi tiết dựa vào tri thức (chỉ đưa ra link http và ký tự "\\n" nếu có trong tri thức của MISA)\nChú ý: Nếu câu trả lời không ở trong tri thức MISA, tự trả lời theo tri thức của mình.'}
        ]
        
    st.write('Trả lời:')
    answer = st.empty()
    answer.markdown('')
    
    info = get_document(question)
    index, document = info
    
    context.markdown('# ' + index + '\n\n' + document)
    if st.button('Lấy câu trả lời'):
        st.session_state.message.append({'role': 'system', 'content': f'MISA:\n{document}'})
        st.session_state.message.append({'role': 'user', 'content': question})
        message_container.dataframe(df_history(st.session_state.message), use_container_width = True)
        num_tokens = sum([count_tokens(v['content']) for v in st.session_state.message])
        REPLACE_API_PARAMS['max_tokens'] = int(4000 - num_tokens)
        used = []
        with st.spinner('Đang sinh câu trả lời...'):
            response = ''
            # REPLACE_API_PARAMS.pop('max_tokens')
            for resp in openai.ChatCompletion.create(messages = st.session_state.message, **REPLACE_API_PARAMS):
                num_tokens += 1
                response += resp.choices[0].delta.content if resp.choices[0].delta.get('content') else ''
                response = parse_response(response, used)
                try:
                    answer.markdown(response, unsafe_allow_html = True)
                except:
                    pass
        st.session_state.message.append({'role': 'assistant', 'content': response})
        st.success(f'Đã tạo xong câu trả lời gồm {num_tokens} tokens tiêu tốn {0.02 * num_tokens / 1000}$')

    
if __name__ == '__main__':
    main()

