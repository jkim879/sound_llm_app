import os
import base64
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.font_manager as fm
from audio_recorder_streamlit import audio_recorder

# Font settings for matplotlib
plt.rc('font', family='MalgunGothic')  # For Windows

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

# Sidebar configuration
with st.sidebar:
    st.markdown("### 설정")
    model_option = st.selectbox(
        '모델 선택',
        ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo-0125'],
        key='model_select'
    )
    summary_type = st.radio(
        '요약 유형',
        ['일반 요약', '회의록', '인터뷰 분석', '강의 노트'],
        index=1
    )
    summary_length = st.slider(
        '요약 길이 조정',
        min_value=10, max_value=500, step=10, value=300
    )
    
    # Initialize session state if not exists
    if 'saved_model' not in st.session_state:
        st.session_state['saved_model'] = model_option
        st.session_state['saved_summary_type'] = summary_type
        st.session_state['saved_summary_length'] = summary_length
    
    if st.button("설정 저장"):
        st.session_state['saved_model'] = model_option
        st.session_state['saved_summary_type'] = summary_type
        st.session_state['saved_summary_length'] = summary_length
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

def analyze_emotion_over_time(text, model_option):
    chunks = text.split(". ")
    emotions = []
    scores = []
    for chunk in chunks:
        if chunk.strip():
            response = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": "system", "content": "텍스트의 감정을 분석하여 '긍정', '중립', '부정' 중 하나로 분류하고, -1(매우 부정)에서 1(매우 긍정) 사이의 점수를 함께 제시해주세요. 형식: [감정];[점수]"},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=100
            )
            result = response.choices[0].message.content.strip().split(';')
            if len(result) == 2:
                emotions.append(result[0])
                # 문자열에서 숫자만 추출하여 float로 변환
                score = float(''.join(filter(lambda x: x.isdigit() or x in ['-', '.'], result[1])))
                scores.append(max(min(score, 1.0), -1.0))  # 값을 -1과 1 사이로 제한
            else:
                emotions.append("중립")
                scores.append(0.0)
    return emotions, scores

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
        keywords = extract_keywords(transcribed_text, model_option)
        st.write(keywords)

        # Topic classification
        st.markdown("### 주제 분류")
        topic = classify_topics(transcribed_text, model_option)
        st.write(topic)

        # Emotion analysis button
        st.markdown(emotion_button_style, unsafe_allow_html=True)
        emotion_analysis_container = st.container()
        if st.button("감정 분석 실행"):
            emotion_analysis_container.empty()
            with emotion_analysis_container:
                with st.spinner("감정 분석 중..."):
                    try:
                        # Text emotion analysis
                        text_response = client.chat.completions.create(
                            model=model_option,
                            messages=[
                                {"role": "system", "content": "당신은 텍스트 감정 분석 전문가입니다."},
                                {"role": "user", "content": f"다음 텍스트의 감정을 분석해주세요: {transcribed_text}"}
                            ],
                            max_tokens=100,
                            temperature=0.5
                        )
                        text_emotion = text_response.choices[0].message.content.strip()
                        st.success(f"텍스트 감정 분석 결과: {text_emotion}")

                        # Emotion over time analysis in a new container
                        emotion_container = st.container()
                        with emotion_container:
                            st.markdown("### 시간에 따른 감정 변화")
                            emotions, scores = analyze_emotion_over_time(transcribed_text, model_option)
                            
                            # 감정 분석 설명 추가
                            st.markdown("""
                            #### 감정 분석 해석 방법
                            
                            **1. 그래프 구성**
                            - 상단: 문장별 감정 상태를 텍스트로 표시 (긍정/중립/부정)
                            - 하단: 시간 흐름에 따른 감정 강도를 수치화하여 표시
                            
                            **2. 감정 점수 해석**
                            - 매우 긍정적 (0.7 ~ 1.0): 강한 기쁨, 열정, 만족감
                            - 긍정적 (0.3 ~ 0.7): 일반적인 기쁨, 호의적 감정
                            - 중립적 (-0.3 ~ 0.3): 객관적, 중립적 감정
                            - 부정적 (-0.7 ~ -0.3): 불만족, 걱정, 약한 부정
                            - 매우 부정적 (-1.0 ~ -0.7): 강한 분노, 실망, 슬픔
                            
                            **3. 그래프 패턴 분석**
                            - 상승 추세: 감정이 점차 긍정적으로 변화
                            - 하강 추세: 감정이 점차 부정적으로 변화
                            - 급격한 변화: 특정 시점에서 감정 변화가 큼
                            - 안정적 패턴: 일정한 감정 상태 유지
                            
                            **4. 중립 영역 (회색 영역)**
                            - 감정 강도가 약하거나 중립적인 구간
                            - 일반적인 대화나 객관적 서술에서 자주 나타남
                            """)
                            
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
                            
                            # 한글 폰트 설정
                            plt.rcParams['font.family'] = 'Malgun Gothic'
                            plt.rcParams['axes.unicode_minus'] = False
                            
                            fig.suptitle("감정 분석 결과", fontsize=16)
                            
                            # 감정 텍스트 표시
                            ax1.set_xticks([])
                            ax1.set_yticks([])
                            ax1.axis('off')
                            emotion_text = ' → '.join(emotions)
                            ax1.text(0.5, 0.5, emotion_text, 
                                ha='center', va='center', 
                                wrap=True,
                                fontsize=12)
                            
                            # 감정 점수 그래프
                            ax2.plot(range(len(scores)), scores, 
                                marker='o', 
                                color='#4CAF50', 
                                linewidth=2, 
                                markersize=8)
                            ax2.grid(True, linestyle='--', alpha=0.7)
                            ax2.set_xlabel("문장 순서", fontsize=12)
                            ax2.set_ylabel("감정 점수\n(-1: 부정, 1: 긍정)", fontsize=12)
                            ax2.set_ylim(-1.1, 1.1)
                            
                            # 중립 영역 표시
                            ax2.axhspan(-0.5, 0.5, color='gray', alpha=0.2, label='중립 영역')
                            ax2.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                        # Combined emotion analysis
                        audio_emotion = "긍정적 (추정)"
                        combined_emotion = f"음성: {audio_emotion}, 텍스트: {text_emotion}"
                        st.markdown(f"<div style='padding:10px; background:#E3F2FD; border-radius:5px;'>종합 감정 분석: {combined_emotion}</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"감정 분석 실패: {str(e)}")

        # Summary button
        st.markdown(summary_button_style, unsafe_allow_html=True)
        summary_container = st.container()
        if st.button("요약 시작"):
            summary_container.empty()
            with summary_container:
                with st.spinner("요약 중..."):
                    try:
                        model_name = st.session_state['saved_model']
                        summary_type = st.session_state.get('saved_summary_type', '회의록')
                        summary_length = st.session_state.get('saved_summary_length', 300)
                        
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": f"당신은 {summary_type} 전문가입니다."},
                                {"role": "user", "content": f"""
                                    다음 내용을 {summary_type} 형식으로 요약해주세요:
                                    {transcribed_text}
                                    요약 내용의 길이는 반드시 {summary_length}에서 해결하세요.
                                    대답은 반드시 존댓말로 해주세요.
                                """}
                            ],
                            max_tokens=summary_length,
                            temperature=0.7
                        )
                        summary = response.choices[0].message.content.strip()
                        st.markdown(f"<div style='padding:10px; background:#E8F5E9; border-radius:5px;'>{summary}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"요약 실패: {str(e)}")

    except Exception as e:
        st.error(f"파일 처리 실패: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>문의: kjs_1000@naver.com </p>", unsafe_allow_html=True)
