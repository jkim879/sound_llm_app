import os
import base64
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.font_manager as fm
from audio_recorder_streamlit import audio_recorder

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# Page configuration
st.set_page_config(
    page_title="음성 요약 분석 서비스",
    layout="wide"
)

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if not client.api_key:
    st.error("API 키가 설정되지 않았습니다")
    st.stop()

# Initialize session state for settings if not exists
if 'settings' not in st.session_state:
    st.session_state['settings'] = {
        'model': 'gpt-4o-mini',
        'summary_type': '회의록',
        'summary_length': 300
    }

# Sidebar configuration
with st.sidebar:
    st.markdown("### 설정")
    temp_model = st.selectbox(
        '모델 선택',
        ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo-0125'],
        key='temp_model',
        index=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo-0125'].index(st.session_state['settings']['model'])
    )
    temp_summary_type = st.radio(
        '요약 유형',
        ['일반 요약', '회의록', '인터뷰 분석', '강의 노트'],
        index=['일반 요약', '회의록', '인터뷰 분석', '강의 노트'].index(st.session_state['settings']['summary_type'])
    )
    temp_summary_length = st.slider(
        '요약 길이 조정',
        min_value=10, max_value=500, step=10,
        value=st.session_state['settings']['summary_length']
    )
    
    if st.button("설정 저장"):
        st.session_state['settings']['model'] = temp_model
        st.session_state['settings']['summary_type'] = temp_summary_type
        st.session_state['settings']['summary_length'] = temp_summary_length
        st.success("설정이 저장되었습니다.")

    st.markdown("---")
    st.markdown("#### 도움말")
    st.info("음성 파일을 업로드하고 요약 및 감정 분석을 시작하세요.")

# Main title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>음성 요약 분석 서비스</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>음성 파일을 텍스트로 변환하고 요약 및 감정을 분석합니다.</p>", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader(
    "음성 파일 선택 (MP3, WAV, M4A)",
    type=["mp3", "wav", "m4a"],
    help="최대 50MB까지 업로드 가능"
)

# Helper functions
def extract_keywords(text, model_option):
    response = client.chat.completions.create(
        model=model_option,
        messages=[
            {"role": "system", "content": "이 대화의 가장 핵심 키워드 추출. 답변은 부가 설명 없이 핵심 키워드만 추출."},
            {"role": "user", "content": text}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def classify_topics(text, model_option):
    response = client.chat.completions.create(
        model=model_option,
        messages=[
            {"role": "system", "content": "텍스트 주제 분류. 답변은 부가설명 없이 주제만 추출."},
            {"role": "user", "content": text}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

if uploaded_file:
    try:
        audio_bytes = uploaded_file.read()
        
        st.markdown("### 음성 파일 재생")
        st.audio(audio_bytes)
        
        # WaveSurfer visualization
        audio_b64 = base64.b64encode(audio_bytes).decode()
        components.html(
            f"""
            <script src="https://unpkg.com/wavesurfer.js@7"></script>
            <div id="waveform"></div>
            <script>
                const audioData = "data:audio/wav;base64,{audio_b64}";
                let wavesurfer = WaveSurfer.create({{
                    container: '#waveform',
                    waveColor: '#4CAF50',
                    progressColor: '#1976D2',
                    height: 100,
                    barWidth: 2,
                    barGap: 1,
                    responsive: true,
                    normalize: true,
                    interact: true,
                    cursorWidth: 1,
                    cursorColor: '#333',
                    autoScroll: true,
                    hideScrollbar: false
                }})
                wavesurfer.load(audioData)
            </script>
            """,
            height=150
        )
        
        # Transcription
        with st.spinner("음성 변환 중..."):
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=uploaded_file,
                response_format="text"
            )
            transcribed_text = transcription.strip()
        
        # Display transcribed text
        st.markdown("<h3 style='text-align: left; font-size: 24px; font-weight: bold;'>텍스트 원문</h3>", unsafe_allow_html=True)
        st.text_area("", value=transcribed_text, height=200)

        # Keyword extraction
        st.markdown("### 키워드 추출")
        keywords = extract_keywords(transcribed_text, st.session_state['settings']['model'])
        st.write(keywords)

        # Topic classification
        st.markdown("### 주제 분류")
        topic = classify_topics(transcribed_text, st.session_state['settings']['model'])
        st.write(topic)

    except Exception as e:
        st.error(f"파일 처리 실패: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>문의: kjs_1000@naver.com </p>", unsafe_allow_html=True)
