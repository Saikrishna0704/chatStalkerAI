"""
🕵️ ChatStalkerAI
A fun RAG + Analytics app for WhatsApp group chat analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.parser import parse_whatsapp_chat, get_participants, get_chat_stats
from utils.analytics import count_word
from utils.embeddings import ChatRAG

# Page config
st.set_page_config(
    page_title="ChatStalkerAI",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark neon theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ff88 !important;
        text-shadow: 0 0 10px #00ff8855;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid #00ff8833;
    }
    
    /* Cards/containers */
    .stMetric {
        background: rgba(0, 255, 136, 0.05);
        border: 1px solid #00ff8833;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 255, 136, 0.1);
        border-radius: 8px;
        color: #00ff88;
        border: 1px solid #00ff8833;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 136, 0.3) !important;
        border: 1px solid #00ff88 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(0, 255, 136, 0.05);
        border: 1px solid #00ff8855;
        color: #ffffff;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ff88;
        box-shadow: 0 0 10px #00ff8855;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: #0a0a0f;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px #00ff8877;
        transform: translateY(-2px);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(0, 255, 136, 0.05);
        border: 1px solid #00ff8855;
        border-radius: 8px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(0, 255, 136, 0.05);
        border: 2px dashed #00ff8855;
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888 !important;
    }
    
    /* Chat messages */
    .chat-message {
        background: rgba(0, 255, 136, 0.08);
        border-left: 3px solid #00ff88;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Result box */
    .result-box {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
    }
    
    .result-number {
        font-size: 4rem;
        font-weight: bold;
        color: #00ff88;
        text-shadow: 0 0 20px #00ff8877;
    }
    
    /* Divider */
    hr {
        border-color: #00ff8833;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'chat_df' not in st.session_state:
        st.session_state.chat_df = None
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def render_sidebar():
    """Render the sidebar with upload and settings."""
    with st.sidebar:
        st.markdown("## 🕵️ ChatStalkerAI")
        st.markdown("*Expose your group chat secrets*")
        st.markdown("---")
        
        # File upload
        st.markdown("### 📁 Upload Chat")
        uploaded_file = st.file_uploader(
            "Drop your WhatsApp export here",
            type=['txt'],
            help="Export your WhatsApp chat as .txt file"
        )
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            df = parse_whatsapp_chat(content)
            
            if not df.empty:
                st.session_state.chat_df = df
                st.success(f"✅ Loaded {len(df)} messages!")
                
                # Show stats
                stats = get_chat_stats(df)
                st.markdown("---")
                st.markdown("### 📊 Chat Stats")
                st.metric("Messages", stats['total_messages'])
                st.metric("Participants", stats['participants'])
                st.caption(f"📅 {stats['date_range']}")
            else:
                st.error("❌ Couldn't parse the chat. Make sure it's a valid WhatsApp export.")
        
        st.markdown("---")
        
        # API Key - Check secrets first, then allow manual input
        api_key = None
        
        # Try to get API key from secrets first
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            api_key = st.secrets['GEMINI_API_KEY']
            if not api_key or api_key == "your-gemini-api-key-here":
                api_key = None
        
        # If no valid key in secrets, show input field
        if not api_key:
            st.markdown("### 🔑 Gemini API Key")
            api_key = st.text_input(
                "Enter your API key",
                type="password",
                help="Required for AI Chat Assistant"
            )
        
        if api_key and api_key != "your-gemini-api-key-here":
            try:
                st.session_state.rag = ChatRAG(api_key)
                if st.session_state.chat_df is not None:
                    st.session_state.rag.load_chat(st.session_state.chat_df)
            except Exception as e:
                st.error(f"❌ Invalid API key: {e}")
        
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
            "Made with 💚 and curiosity"
            "</div>",
            unsafe_allow_html=True
        )


def render_chat_assistant():
    """Render the RAG chat assistant tab."""
    st.markdown("## 💬 Chat Assistant")
    st.markdown("*Ask me anything about your group chat!*")
    
    if st.session_state.chat_df is None:
        st.warning("👈 Upload a WhatsApp chat export first!")
        return
    
    if st.session_state.rag is None:
        st.warning("👈 Enter your Gemini API key to enable the chat assistant!")
        return
    
    # Make sure RAG has the chat loaded
    st.session_state.rag.load_chat(st.session_state.chat_df)
    
    # Example queries
    st.markdown("**💡 Try asking:**")
    examples = [
        "Summarize what everyone talked about",
        "Who talks the most about food?",
        "What was the funniest moment?",
        "Who seems to be the group leader?",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"🔮 {example}", key=f"example_{i}"):
                st.session_state.pending_query = example
    
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "🔍 Ask a question",
        placeholder="e.g., What did everyone say about the party?",
        key="query_input"
    )
    
    # Check for pending query from example buttons
    if hasattr(st.session_state, 'pending_query'):
        query = st.session_state.pending_query
        del st.session_state.pending_query
    
    if query:
        with st.spinner("🔍 Stalking the chat..."):
            response = st.session_state.rag.query(query)
        
        st.markdown("### 🕵️ Here's what I found:")
        st.markdown(
            f"<div class='chat-message'>{response}</div>",
            unsafe_allow_html=True
        )
    
    # Quick summary button
    st.markdown("---")
    if st.button("📋 Get Chat Summary", use_container_width=True):
        with st.spinner("🔍 Analyzing the whole chat..."):
            summary = st.session_state.rag.get_summary()
        st.markdown("### 📋 Chat Summary")
        st.markdown(
            f"<div class='chat-message'>{summary}</div>",
            unsafe_allow_html=True
        )


def render_word_counter():
    """Render the word counter analytics tab."""
    st.markdown("## 📊 Word Counter")
    st.markdown("*Find out who says what the most!*")
    
    if st.session_state.chat_df is None:
        st.warning("👈 Upload a WhatsApp chat export first!")
        return
    
    df = st.session_state.chat_df
    participants = get_participants(df)
    
    # Three columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Block 1: Word input
        word = st.text_input(
            "🔤 Enter a word to search",
            placeholder="e.g., lol, pizza, meeting",
            help="Search for any word or phrase"
        )
    
    with col2:
        # Block 2: Participant dropdown
        participant_options = ["All"] + participants
        selected_participant = st.selectbox(
            "👤 Select Participant",
            options=participant_options,
            help="Filter by participant or search all"
        )
    
    st.markdown("---")
    
    # Block 3: Results
    if word:
        result = count_word(df, word, selected_participant if selected_participant != "All" else None)
        
        if selected_participant == "All":
            # Show total and bar chart
            st.markdown(
                f"""
                <div class='result-box'>
                    <div style='color: #888; margin-bottom: 10px;'>Total occurrences of "{word}"</div>
                    <div class='result-number'>{result['total']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if result['by_participant']:
                # Create bar chart
                chart_data = pd.DataFrame([
                    {"Participant": k, "Count": v} 
                    for k, v in result['by_participant'].items()
                ])
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=chart_data['Count'],
                    y=chart_data['Participant'],
                    orientation='h',
                    marker=dict(
                        color=chart_data['Count'],
                        colorscale=[[0, '#004d40'], [0.5, '#00ff88'], [1, '#00ffcc']],
                        line=dict(color='#00ff88', width=1)
                    ),
                    text=chart_data['Count'],
                    textposition='outside',
                    textfont=dict(color='#00ff88', size=14)
                ))
                
                fig.update_layout(
                    title=dict(
                        text=f'Who says "{word}" the most?',
                        font=dict(color='#00ff88', size=18)
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(0,255,136,0.1)',
                        title=dict(text='Count', font=dict(color='#888'))
                    ),
                    yaxis=dict(
                        showgrid=False,
                        title=dict(text=''),
                        autorange='reversed'
                    ),
                    margin=dict(l=20, r=20, t=60, b=40),
                    height=max(300, len(chart_data) * 40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"🤷 No one has used '{word}' in the chat!")
        
        else:
            # Show just the count for selected participant
            count = result['by_participant'].get(selected_participant, 0)
            
            st.markdown(
                f"""
                <div class='result-box'>
                    <div style='color: #888; margin-bottom: 10px;'>
                        {selected_participant} said "{word}"
                    </div>
                    <div class='result-number'>{count}</div>
                    <div style='color: #888; margin-top: 10px;'>times</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if count == 0:
                st.info(f"🤷 {selected_participant} hasn't used '{word}' in the chat!")
    else:
        # Show placeholder
        st.markdown(
            """
            <div style='text-align: center; padding: 50px; color: #666;'>
                <div style='font-size: 3rem;'>🔍</div>
                <div style='margin-top: 10px;'>Enter a word above to start stalking!</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """Main app entry point."""
    init_session_state()
    render_sidebar()
    
    # Main content area
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 3rem;'>
            🕵️ ChatStalkerAI
        </h1>
        <p style='text-align: center; color: #888; margin-bottom: 30px;'>
            Your group chat has no secrets anymore
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Tabs
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Word Counter"])
    
    with tab1:
        render_chat_assistant()
    
    with tab2:
        render_word_counter()


if __name__ == "__main__":
    main()

