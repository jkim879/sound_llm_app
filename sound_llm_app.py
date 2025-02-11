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
        'summary_type': '일반 요약',
        'summary_length': 1000
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
        min_value=10, max_value=1000, step=10,
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
            {"role": "system", "content": "다음 내용은 기업신용평가회사 '이크레더블'이 고객에게 전화를 먼저 걸고 대화를 진행한 상황에 대한 내용입니다. 전체 대화를 고려하여 문맥에 맞게 이 대화의 가장 핵심 키워드를 태그해주세요. 답변은 부가 설명 없이 핵심 키워드만 추출."},
            {"role": "user", "content": text}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def analyze_emotion_over_time(text, model_option):
    chunks = text.split(". ")
    emotions = []
    scores = []
    for chunk in chunks:
        if chunk.strip():
            response = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": "system", "content": "다음 내용은 기업신용평가회사 '이크레더블'이 고객에게 전화를 먼저 걸고 대화를 진행한 상황에 대한 내용입니다. 전체상황을 고려하여 문맥에 맞게 고객의 감정만 분석하여 '긍정', '중립', '부정' 중 하나로 분류하고, -1(매우 부정)에서 1(매우 긍정) 사이의 점수를 함께 제시해주세요. 형식: [감정];[점수]"},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=1000
            )
            result = response.choices[0].message.content.strip().split(';')
            if len(result) == 2:
                emotions.append(result[0])
                score = float(''.join(filter(lambda x: x.isdigit() or x in ['-', '.'], result[1])))
                scores.append(max(min(score, 1.0), -1.0))
            else:
                emotions.append("중립")
                scores.append(0.0)
    return emotions, scores

def classify_topics(text, model_option):
    response = client.chat.completions.create(
        model=model_option,
        messages=[
            {"role": "system", "content": "다음 내용은 기업신용평가회사 '이크레더블'이 고객에게 전화를 먼저 걸고 대화를 진행한 상황에 대한 내용입니다. 문맥에 맞게 텍스트 주제 분류. 답변은 부가설명 없이 주제만 추출."},
            {"role": "user", "content": text}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# Custom button styles
emotion_button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #2196F3;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1976D2;
    }
    </style>
"""

summary_button_style = """
    <style>
    div.stButton > button:last-child {
        background-color: #26A69A;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        border: none;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    div.stButton > button:hover {
        background-color: #00796B;
    }
    </style>
"""

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
                model="openai/whisper-large-v2",
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

        # Initialize session state for visibility toggles if not exists
        if 'show_emotion_result' not in st.session_state:
            st.session_state['show_emotion_result'] = False
        if 'show_summary_result' not in st.session_state:
            st.session_state['show_summary_result'] = False
        if 'emotion_result' not in st.session_state:
            st.session_state['emotion_result'] = None
        if 'summary_result' not in st.session_state:
            st.session_state['summary_result'] = None

        # Emotion analysis button and result
        st.markdown(emotion_button_style, unsafe_allow_html=True)
        if st.button("감정 분석 실행"):
            with st.spinner("감정 분석 중..."):
                try:
                    text_response = client.chat.completions.create(
                        model=st.session_state['settings']['model'],
                        messages=[
                            {"role": "system", "content": "당신은 텍스트 감정 분석 전문가입니다."},
                            {"role": "user", "content": f"다음 내용은 기업신용평가회사 '이크레더블'이 고객에게 전화를 먼저 걸고 대화를 진행한 상황에 대한 내용입니다. 다음 텍스트에서 전체 상황을 고려하여 문맥에 맞게 고객의 감정만 분석해주세요: {transcribed_text}. 대답은 반드시 요약 형식으로 답변."}
                        ],
                        max_tokens=1000,
                        temperature=0.5
                    )
                    st.session_state['emotion_result'] = text_response.choices[0].message.content.strip()
                    st.session_state['show_emotion_result'] = True
                except Exception as e:
                    st.error(f"감정 분석 실패: {str(e)}")

        # Show/hide emotion analysis result
        if st.session_state['show_emotion_result'] and st.session_state['emotion_result']:
            with st.expander("감정 분석 결과", expanded=True):
                st.markdown(f"<div style='padding:10px; background:#E8F5E9; border-radius:5px;'>{st.session_state['emotion_result']}</div>", unsafe_allow_html=True)

        # Summary button and result
        st.markdown(summary_button_style, unsafe_allow_html=True)
        if st.button("요약 시작"):
            with st.spinner("요약 중..."):
                try:
                    response = client.chat.completions.create(
                        model=st.session_state['settings']['model'],
                        messages=[
                            {"role": "system", "content": f"당신은 {st.session_state['settings']['summary_type']} 전문가입니다."},
                            {"role": "user", "content": f"""
                                다음 내용은 기업신용평가회사 '이크레더블'이 고객에게 전화를 먼저 걸고 대화를 진행한 상황에 대한 내용입니다.
                                다음 내용을 {st.session_state['settings']['summary_type']} 형식으로 문맥에 맞게 요약해주세요:
                                {transcribed_text}
                                요약 내용의 길이는 반드시 {st.session_state['settings']['summary_length']}에서 해결하세요.
                                대답은 반드시 요약 형식으로 답변.
                                전체적인 내용을 한 번 더 파악해서 오타나 오탈자 수정.
                                인물에 집중하지 말고 상황에 집중해서 가독성 좋게 보고서 형식으로 요약. 1,2,3,4 사용해서 맥락 구분 중요.
                                불필요한 내용 넣지말고 간략하게 요약.
                            """}
                        ],
                        max_tokens=st.session_state['settings']['summary_length'],
                        temperature=0.7
                    )
                    st.session_state['summary_result'] = response.choices[0].message.content.strip()
                    st.session_state['show_summary_result'] = True
                except Exception as e:
                    st.error(f"요약 실패: {str(e)}")

        # Show/hide summary result
        if st.session_state['show_summary_result'] and st.session_state['summary_result']:
            with st.expander("요약 결과", expanded=True):
                st.markdown(f"<div style='padding:10px; background:#E8F5E9; border-radius:5px;'>{st.session_state['summary_result']}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"파일 처리 실패: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>문의: kjs_1000@naver.com </p>", unsafe_allow_html=True)
