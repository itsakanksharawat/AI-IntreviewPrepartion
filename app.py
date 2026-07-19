import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import random
import json
from datetime import datetime, timedelta
import numpy as np

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Interview Prep",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI Interview Prep - Your personal coding interview coach"
    }
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --light-bg: #f8fafc;
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom metrics */
    [data-testid="metric-container"] {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #6366f1;
        margin: 10px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3);
    }
    
    /* Success, warning, error messages */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        color: #92400e !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 30px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_resource
def load_question_bank():
    """Load questions from JSON file"""
    try:
        with open("question.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        # Return default questions if file not found
        return {
            "Arrays": [
                "Find duplicate element in array",
                "Maximum subarray sum",
                "Rotate array by k steps",
                "Two Sum Problem",
                "Merge two sorted arrays",
                "Product of Array Except Self",
                "Container With Most Water"
            ],
            "Linked List": [
                "Reverse Linked List",
                "Detect cycle in linked list",
                "Merge two linked lists",
                "Find middle node",
                "Add Two Numbers",
                "LRU Cache Design",
                "Copy List with Random Pointer"
            ],
            "Stack": [
                "Valid Parentheses",
                "Implement stack using queue",
                "Next Greater Element",
                "Min Stack",
                "Postfix Evaluation",
                "Trapping Rain Water",
                "Largest Rectangle in Histogram"
            ],
            "Queue": [
                "Implement Queue using Stacks",
                "Number of Recent Calls",
                "Moving Average from Data Stream",
                "Sliding Window Maximum",
                "First Unique Character in String"
            ],
            "DBMS": [
                "Design a database schema",
                "Write complex SQL joins",
                "Explain normalization",
                "Index optimization",
                "Transaction handling"
            ],
            "Operating Systems": [
                "Explain process vs thread",
                "CPU scheduling algorithms",
                "Deadlock prevention",
                "Memory management",
                "Virtual memory concepts"
            ],
            "System Design": [
                "Design Twitter",
                "Design Uber",
                "Design Netflix",
                "Design a URL shortener",
                "Design a chat system"
            ]
        }

# ==================== SESSION STATE ====================
if 'scores' not in st.session_state:
    st.session_state.scores = {}

if 'profile' not in st.session_state:
    st.session_state.profile = {}

if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []

if 'study_plan' not in st.session_state:
    st.session_state.study_plan = {}

# ==================== HELPER FUNCTIONS ====================

def calculate_readiness(avg_score):
    """Predict interview readiness using ML model"""
    X = [[20], [35], [45], [55], [65], [75], [85], [95]]
    y = ['Low', 'Low', 'Medium', 'Medium', 'Good', 'Good', 'Excellent', 'Excellent']
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    prediction = model.predict([[avg_score]])
    return prediction[0]

def get_readiness_color(readiness):
    """Get color based on readiness level"""
    colors = {
        'Low': '#ef4444',
        'Medium': '#f59e0b',
        'Good': '#3b82f6',
        'Excellent': '#10b981'
    }
    return colors.get(readiness, '#6366f1')

def get_readiness_emoji(readiness):
    """Get emoji based on readiness level"""
    emojis = {
        'Low': '🔴',
        'Medium': '🟡',
        'Good': '🔵',
        'Excellent': '🟢'
    }
    return emojis.get(readiness, '⚪')

def create_score_chart(scores):
    """Create interactive score chart"""
    df = pd.DataFrame({
        'Topic': list(scores.keys()),
        'Score': list(scores.values())
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Topic'],
            y=df['Score'],
            marker=dict(
                color=df['Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=df['Score'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Topic-wise Performance",
        xaxis_title="Topics",
        yaxis_title="Score (%)",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_radar_chart(scores):
    """Create radar chart for skill visualization"""
    fig = go.Figure(data=go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself',
        name='Your Skills'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_progress_chart(scores):
    """Create progress tracking chart"""
    df = pd.DataFrame({
        'Topic': list(scores.keys()),
        'Current': list(scores.values()),
        'Target': [100] * len(scores)
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Score',
        x=df['Topic'],
        y=df['Current'],
        marker_color='#6366f1'
    ))
    
    fig.add_trace(go.Bar(
        name='Target Score',
        x=df['Topic'],
        y=df['Target'],
        marker_color='#e2e8f0'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Score vs Target",
        xaxis_title="Topics",
        yaxis_title="Score (%)",
        height=400,
        template="plotly_white",
        hovermode='group'
    )
    
    return fig

def generate_study_plan(weak_topics):
    """Generate personalized study plan"""
    study_plans = {
        "Arrays": {
            "Week 1": ["Array Basics", "Indexing and Slicing", "Basic Traversal"],
            "Week 2": ["Sliding Window Technique", "Two Pointer Approach", "Prefix Sum"],
            "Week 3": ["LeetCode Easy Problems", "Pattern Recognition", "Time Complexity"],
            "Week 4": ["LeetCode Medium Problems", "Mock Interviews", "Optimization"]
        },
        "Linked List": {
            "Week 1": ["Linked List Basics", "Node Creation", "Basic Traversal"],
            "Week 2": ["Reverse Operations", "Fast & Slow Pointers", "Cycle Detection"],
            "Week 3": ["Merge Operations", "LeetCode Medium", "Common Patterns"],
            "Week 4": ["Advanced Problems", "System Design with LL", "Mock Interviews"]
        },
        "Stack": {
            "Week 1": ["Stack Basics", "Push & Pop Operations", "LIFO Principle"],
            "Week 2": ["Monotonic Stack", "Expression Evaluation", "Parenthesis Problems"],
            "Week 3": ["LeetCode Medium", "Stack with Queue", "Advanced Patterns"],
            "Week 4": ["Complex Problems", "Optimization Techniques", "Mock Interviews"]
        },
        "Queue": {
            "Week 1": ["Queue Basics", "FIFO Principle", "Basic Operations"],
            "Week 2": ["Circular Queue", "Deque", "BFS Applications"],
            "Week 3": ["Priority Queue", "LeetCode Medium", "Level Order Traversal"],
            "Week 4": ["Complex Problems", "System Design", "Mock Interviews"]
        },
        "DBMS": {
            "Week 1": ["SQL Basics", "SELECT Queries", "WHERE Clauses"],
            "Week 2": ["JOINs (INNER, LEFT, RIGHT)", "Aggregation", "GROUP BY"],
            "Week 3": ["Normalization", "Keys & Constraints", "Indexing"],
            "Week 4": ["Transactions", "Views", "Stored Procedures"]
        },
        "Operating Systems": {
            "Week 1": ["Process Basics", "Threads", "Context Switching"],
            "Week 2": ["CPU Scheduling", "Synchronization", "Mutex & Semaphore"],
            "Week 3": ["Deadlock", "Memory Management", "Virtual Memory"],
            "Week 4": ["File Systems", "I/O Management", "System Calls"]
        },
        "System Design": {
            "Week 1": ["Scalability Basics", "Load Balancing", "Database Sharding"],
            "Week 2": ["Caching Strategies", "CDN", "Message Queues"],
            "Week 3": ["Microservices", "API Design", "Monitoring"],
            "Week 4": ["Real System Design", "Trade-offs", "Mock System Design"]
        }
    }
    
    return study_plans

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🎯 AI Interview Prep")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "👤 Student Profile",
            "📊 Skill Assessment",
            "📈 Dashboard",
            "🛣️ AI Roadmap",
            "❓ Interview Questions",
            "📉 Progress Tracker",
            "⚙️ Settings"
        ]
    )
    
    st.markdown("---")
    
    if st.session_state.profile:
        st.success(f"👋 Welcome, {st.session_state.profile.get('name', 'Student')}")
    
    st.markdown("---")
    st.markdown("""
    ### 📚 Features
    - Smart skill assessment
    - Personalized learning paths
    - Interview question generator
    - Progress tracking
    - AI-powered insights
    """)

# ==================== HOME PAGE ====================
if menu == "🏠 Home":
    st.title("🚀 AI Interview Prep")
    st.markdown("""
    ### Master Technical Interviews with AI-Powered Insights
    
    Welcome to your personal coding interview coach! This platform helps you:
    - Assess your coding skills accurately
    - Identify weak areas with precision
    - Get personalized learning roadmaps
    - Practice interview questions
    - Track your progress in real-time
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👥 Active Users", "5000+", "+500 this month")
    
    with col2:
        st.metric("❓ Questions Available", "1000+", "+100 new")
    
    with col3:
        st.metric("✅ Success Rate", "92%", "+5% this quarter")
    
    st.markdown("---")
    
    st.subheader("🎯 What You Can Do Here")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Assessment & Analytics
        - **📊 Skill Assessment** - Rate yourself on 10+ topics
        - **⚠️ Weak Topic Detection** - AI identifies your problem areas
        - **📈 Performance Analytics** - Visualize your progress
        """)
    
    with col2:
        st.markdown("""
        ### Learning & Growth
        - **🛣️ AI Learning Roadmap** - Personalized study plans
        - **❓ Interview Questions** - Practice with real questions
        - **📉 Progress Tracker** - Monitor improvement over time
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👤 Create Profile", use_container_width=True, key="home_profile"):
            st.switch_page("pages/student_profile.py")
    
    with col2:
        if st.button("📊 Start Assessment", use_container_width=True, key="home_assessment"):
            st.switch_page("pages/skill_assessment.py")
    
    with col3:
        if st.button("📈 View Dashboard", use_container_width=True, key="home_dashboard"):
            st.switch_page("pages/dashboard.py")
    
    st.markdown("---")
    
    st.info("""
    💡 **Pro Tip:** Start by creating your profile, then take the skill assessment 
    to get a personalized learning roadmap!
    """)

# ==================== STUDENT PROFILE ====================
elif menu == "👤 Student Profile":
    st.title("🎓 Student Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value=st.session_state.profile.get('name', ''))
            email = st.text_input("Email", value=st.session_state.profile.get('email', ''))
        
        with col2:
            branch = st.selectbox(
                "Branch",
                ["BCA", "BTech", "MCA", "BSc(IT)", "MSc(IT)"],
                index=["BCA", "BTech", "MCA", "BSc(IT)", "MSc(IT)"].index(st.session_state.profile.get('branch', 'BCA'))
            )
            year = st.selectbox(
                "Year",
                ["1st Year", "2nd Year", "3rd Year", "Final Year", "Passout"],
                index=["1st Year", "2nd Year", "3rd Year", "Final Year", "Passout"].index(st.session_state.profile.get('year', '1st Year'))
            )
        
        role = st.selectbox(
            "Target Role",
            ["Frontend Developer", "Backend Developer", "Full Stack Developer", 
             "Data Analyst", "ML Engineer", "AI Engineer", "Data Scientist"],
            index=["Frontend Developer", "Backend Developer", "Full Stack Developer", 
                   "Data Analyst", "ML Engineer", "AI Engineer", "Data Scientist"].index(
                st.session_state.profile.get('role', 'Full Stack Developer'))
        )
        
        experience = st.slider("Years of Experience", 0, 10, 
                               st.session_state.profile.get('experience', 0))
        
        skills = st.multiselect(
            "Current Skills",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "React", "Angular"],
            default=st.session_state.profile.get('skills', [])
        )
        
        bio = st.text_area("About You", value=st.session_state.profile.get('bio', ''))
        
        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        
        if submitted:
            st.session_state.profile = {
                "name": name,
                "email": email,
                "role": role,
                "branch": branch,
                "year": year,
                "experience": experience,
                "skills": skills,
                "bio": bio,
                "created_at": datetime.now().isoformat()
            }
            st.success("✅ Profile saved successfully!")
            st.balloons()
    
    # Display saved profile
    if st.session_state.profile:
        st.markdown("---")
        st.subheader("📋 Your Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **👤 Name:** {st.session_state.profile.get('name', 'N/A')}
            
            **📧 Email:** {st.session_state.profile.get('email', 'N/A')}
            
            **💼 Role:** {st.session_state.profile.get('role', 'N/A')}
            
            **🎓 Branch:** {st.session_state.profile.get('branch', 'N/A')}
            """)
        
        with col2:
            st.markdown(f"""
            **📅 Year:** {st.session_state.profile.get('year', 'N/A')}
            
            **💻 Experience:** {st.session_state.profile.get('experience', 0)} years
            
            **🏆 Skills:** {', '.join(st.session_state.profile.get('skills', []))}
            """)

# ==================== SKILL ASSESSMENT ====================
elif menu == "📊 Skill Assessment":
    st.title("🔍 Skill Assessment")
    
    st.markdown("""
    Rate your proficiency level for each topic (0-100).
    Be honest! This helps us create a personalized learning path.
    """)
    
    topics = [
        "Arrays",
        "Linked List",
        "Stack",
        "Queue",
        "DBMS",
        "Operating Systems",
        "Programming Languages",
        "Data Structures",
        "Algorithms",
        "System Design"
    ]
    
    scores_input = {}
    
    # Create columns for better layout
    cols = st.columns(2)
    
    for idx, topic in enumerate(topics):
        col = cols[idx % 2]
        with col:
            scores_input[topic] = st.slider(
                topic,
                0, 100, 
                st.session_state.scores.get(topic, 50),
                step=5,
                key=f"slider_{topic}"
            )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Scores", use_container_width=True):
            st.session_state.scores = scores_input
            st.session_state.assessment_history.append({
                'date': datetime.now().isoformat(),
                'scores': scores_input.copy()
            })
            st.success("✅ Scores saved successfully!")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.scores = {}
            st.rerun()
    
    # Show stats
    if st.session_state.scores:
        st.markdown("---")
        
        avg_score = sum(st.session_state.scores.values()) / len(st.session_state.scores)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Average Score", f"{avg_score:.1f}%")
        
        with col2:
            weak_count = len([s for s in st.session_state.scores.values() if s < 50])
            st.metric("⚠️ Weak Topics", weak_count)
        
        with col3:
            strong_count = len([s for s in st.session_state.scores.values() if s >= 75])
            st.metric("✅ Strong Topics", strong_count)
        
        with col4:
            readiness = calculate_readiness(avg_score)
            st.metric("🎯 Readiness", readiness)
        
        # Visualization
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_score_chart(st.session_state.scores), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_radar_chart(st.session_state.scores), use_container_width=True)
        
        # Feedback
        st.markdown("---")
        st.subheader("🤖 AI Feedback")
        
        if avg_score >= 80:
            st.success("🌟 Excellent preparation level! You're interview ready.")
        elif avg_score >= 60:
            st.info("👍 Good progress! Focus on weak topics to improve further.")
        elif avg_score >= 40:
            st.warning("⚡ You're on the right track. Consistent practice will help.")
        else:
            st.error("💪 Start with fundamentals. You'll improve with dedicated practice.")

# ==================== DASHBOARD ====================
elif menu == "📈 Dashboard":
    st.title("📊 Dashboard")
    
    if not st.session_state.profile:
        st.warning("⚠️ Please complete your profile first!")
        st.info("Go to **Student Profile** to get started.")
    elif not st.session_state.scores:
        st.warning("⚠️ Please complete your skill assessment!")
        st.info("Go to **Skill Assessment** to evaluate your knowledge.")
    else:
        profile = st.session_state.profile
        scores = st.session_state.scores
        
        # Student Info
        st.subheader("👨‍🎓 Student Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👤 Name", profile.get('name', 'N/A'))
        with col2:
            st.metric("💼 Role", profile.get('role', 'N/A'))
        with col3:
            st.metric("🎓 Branch", profile.get('branch', 'N/A'))
        with col4:
            st.metric("📅 Year", profile.get('year', 'N/A'))
        
        st.markdown("---")
        
        # Key Metrics
        st.subheader("📊 Performance Metrics")
        
        avg_score = sum(scores.values()) / len(scores)
        weak_topics = [topic for topic, score in scores.items() if score < 50]
        strong_topics = [topic for topic, score in scores.items() if score >= 75]
        readiness = calculate_readiness(avg_score)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Average Score", f"{avg_score:.1f}%", 
                     delta=f"{avg_score - (st.session_state.assessment_history[-2]['scores'] if len(st.session_state.assessment_history) > 1 else {}).get('avg', 0) if st.session_state.assessment_history else 0:.1f}%")
        
        with col2:
            st.metric("⚠️ Weak Topics", len(weak_topics))
        
        with col3:
            st.metric("✅ Strong Topics", len(strong_topics))
        
        with col4:
            color = get_readiness_color(readiness)
            st.metric(f"{get_readiness_emoji(readiness)} Readiness", readiness)
        
        st.markdown("---")
        
        # Charts
        st.subheader("📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_score_chart(scores), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_radar_chart(scores), use_container_width=True)
        
        st.markdown("---")
        
        # Topics Table
        st.subheader("📚 Topic Performance Breakdown")
        
        df = pd.DataFrame({
            'Topic': list(scores.keys()),
            'Score': list(scores.values()),
            'Status': ['✅' if score >= 75 else '⚠️' if score >= 50 else '❌' 
                      for score in scores.values()]
        })
        
        st.dataframe(df.set_index('Topic'), use_container_width=True)
        
        st.markdown("---")
        
        # Weak Topics
        st.subheader("⚠️ Areas for Improvement")
        
        if weak_topics:
            col_count = 3
            cols = st.columns(col_count)
            for idx, topic in enumerate(weak_topics):
                with cols[idx % col_count]:
                    st.error(f"**{topic}**\n\nScore: {scores[topic]}%")
        else:
            st.success("🎉 No weak topics! Keep maintaining excellence.")
        
        st.markdown("---")
        
        # Strong Topics
        st.subheader("✅ Your Strengths")
        
        if strong_topics:
            col_count = 3
            cols = st.columns(col_count)
            for idx, topic in enumerate(strong_topics):
                with cols[idx % col_count]:
                    st.success(f"**{topic}**\n\nScore: {scores[topic]}%")
        else:
            st.info("Keep working on your skills to build strong topics!")

# ==================== AI ROADMAP ====================
elif menu == "🛣️ AI Roadmap":
    st.title("🛣️ AI Learning Roadmap")
    
    if not st.session_state.scores:
        st.warning("⚠️ Please complete your skill assessment first!")
        st.info("Go to **Skill Assessment** to get your personalized roadmap.")
    else:
        scores = st.session_state.scores
        weak_topics = [k for k, v in scores.items() if v < 50]
        avg_score = sum(scores.values()) / len(scores)
        
        if not weak_topics:
            st.success("""
            🎉 **Excellent Performance!**
            
            Your skills are well-developed. Continue with:
            - Advanced problem-solving
            - System design concepts
            - Real-world project implementation
            - Mock interviews with professionals
            """)
        else:
            st.markdown(f"### 📚 Personalized Study Plan")
            st.markdown(f"Based on your average score of **{avg_score:.1f}%**, here's your roadmap:\n")
            
            study_plans = generate_study_plan(weak_topics)
            
            for idx, topic in enumerate(weak_topics, 1):
                with st.expander(f"**{idx}. {topic}** (Current: {scores[topic]}%) 📖", expanded=False):
                    
                    if topic in study_plans:
                        plan = study_plans[topic]
                        
                        for week, tasks in plan.items():
                            st.markdown(f"#### {week}")
                            for task in tasks:
                                st.markdown(f"- {task}")
                            st.markdown("")
                    
                    # Progress tracker for this topic
                    st.markdown(f"**Target Score Progress**")
                    progress = scores[topic] / 100
                    st.progress(progress)
                    st.markdown(f"Current: {scores[topic]}/100 | Target: 100/100")
            
            st.markdown("---")
            
            # Study Resources
            st.subheader("📚 Recommended Resources")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### Coding Practice
                - **LeetCode** - Top interview questions
                - **HackerRank** - Algorithm practice
                - **CodeSignal** - Interview simulations
                """)
            
            with col2:
                st.markdown("""
                ### System Design
                - **Grokking the System Design**
                - **System Design Interview Book**
                - **YouTube: Tech Dummies**
                """)
            
            with col3:
                st.markdown("""
                ### DSA Concepts
                - **GeeksforGeeks** - Theory & examples
                - **Abdul Bari** - YouTube tutorials
                - **Strivers Sheet** - Comprehensive guide
                """)
            
            st.markdown("---")
            
            # Timeline
            st.subheader("⏱️ Suggested Timeline")
            
            weeks_needed = len(weak_topics) * 4
            end_date = datetime.now() + timedelta(weeks=weeks_needed)
            
            st.info(f"""
            📅 **Estimated Preparation Time:** {weeks_needed} weeks
            
            **Start Date:** {datetime.now().strftime('%B %d, %Y')}
            
            **Target Date:** {end_date.strftime('%B %d, %Y')}
            """)

# ==================== INTERVIEW QUESTIONS ====================
elif menu == "❓ Interview Questions":
    st.title("❓ Interview Question Generator")
    
    question_bank = load_question_bank()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        topic = st.selectbox(
            "Select Topic",
            list(question_bank.keys())
        )
    
    with col2:
        difficulty = st.selectbox(
            "Difficulty Level",
            ["Easy", "Medium", "Hard"]
        )
    
    with col3:
        num_questions = st.slider("Number of Questions", 1, 10, 3)
    
    if st.button("🔄 Generate Questions", use_container_width=True):
        available_questions = question_bank.get(topic, [])
        
        if available_questions:
            selected_questions = random.sample(
                available_questions,
                min(num_questions, len(available_questions))
            )
            
            st.markdown("---")
            st.subheader(f"🎯 {difficulty} Level - {topic}")
            
            for i, question in enumerate(selected_questions, 1):
                with st.expander(f"**Q{i}:** {question}", expanded=(i == 1)):
                    st.markdown(f"""
                    ### Question Details
                    
                    **Topic:** {topic}
                    
                    **Difficulty:** {difficulty}
                    
                    **Problem:** {question}
                    
                    ---
                    
                    ### Solution Approach
                    
                    1. **Understanding:** Read and understand the problem
                    2. **Approach:** Think of the best algorithm
                    3. **Implementation:** Code the solution
                    4. **Testing:** Test with examples
                    5. **Optimization:** Optimize time & space
                    
                    **Time Complexity:** To be determined
                    
                    **Space Complexity:** To be determined
                    """)
                    
                    # Additional resources
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"💡 See Hint - Q{i}", key=f"hint_{i}"):
                            st.info("Think about edge cases and optimal data structures!")
                    
                    with col2:
                        if st.button(f"✅ Mark as Done - Q{i}", key=f"done_{i}"):
                            st.success("Great! Continue practicing.")
        else:
            st.error(f"No questions available for {topic}")
    
    st.markdown("---")
    st.subheader("📊 Questions Statistics")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        total_q = sum(len(q) for q in question_bank.values())
        st.metric("📚 Total Questions", total_q)
    
    with stats_col2:
        st.metric("🏷️ Topics Available", len(question_bank))
    
    with stats_col3:
        st.metric("⭐ Difficulty Levels", 3)

# ==================== PROGRESS TRACKER ====================
elif menu == "📉 Progress Tracker":
    st.title("📉 Progress Tracker")
    
    if not st.session_state.scores:
        st.warning("⚠️ Please complete your skill assessment first!")
        st.info("Go to **Skill Assessment** to start tracking your progress.")
    else:
        scores = st.session_state.scores
        
        # Current Progress Table
        st.subheader("📊 Current Performance")
        
        progress_df = pd.DataFrame({
            'Topic': list(scores.keys()),
            'Current Score': list(scores.values()),
            'Target Score': [100] * len(scores),
            'Gap': [100 - score for score in scores.values()]
        })
        
        st.dataframe(
            progress_df.style.format({'Current Score': '{:.0f}%', 'Target Score': '{:.0f}%', 'Gap': '{:.0f}%'}),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Progress Chart
        st.subheader("📈 Progress Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_progress_chart(scores), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_radar_chart(scores), use_container_width=True)
        
        st.markdown("---")
        
        # Interview Readiness Prediction
        st.subheader("🎯 Interview Readiness Analysis")
        
        avg_score = sum(scores.values()) / len(scores)
        readiness = calculate_readiness(avg_score)
        color = get_readiness_color(readiness)
        
        # Custom gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_score,
            title={'text': "Overall Readiness Score"},
            delta={'reference': 80, 'suffix': " from target"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 35], 'color': "#ef4444"},
                    {'range': [35, 60], 'color': "#f59e0b"},
                    {'range': [60, 80], 'color': "#3b82f6"},
                    {'range': [80, 100], 'color': "#10b981"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Readiness Details
        st.subheader("📋 Readiness Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Average Score", f"{avg_score:.1f}%")
        
        with col2:
            st.metric(f"{get_readiness_emoji(readiness)} Status", readiness)
        
        with col3:
            weak_count = len([s for s in scores.values() if s < 50])
            st.metric("⚠️ Weak Topics", weak_count)
        
        with col4:
            strong_count = len([s for s in scores.values() if s >= 75])
            st.metric("✅ Strong Topics", strong_count)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Personalized Recommendations")
        
        if readiness == 'Excellent':
            st.success("""
            🎉 **You're Interview Ready!**
            
            Next Steps:
            1. Apply to your target companies
            2. Do mock interviews with friends/mentors
            3. Practice system design interviews
            4. Review your weak areas one more time
            """)
        
        elif readiness == 'Good':
            st.info("""
            👍 **Good Progress!**
            
            Next Steps:
            1. Focus on your weak topics (< 50%)
            2. Practice LeetCode medium/hard problems
            3. Start system design preparation
            4. Do mock interviews
            """)
        
        elif readiness == 'Medium':
            st.warning("""
            ⚡ **You're on the right track**
            
            Next Steps:
            1. Review DSA fundamentals
            2. Practice more problems (focus on weak topics)
            3. Watch tutorial videos for concepts
            4. Solve 2-3 problems daily
            """)
        
        else:
            st.error("""
            💪 **Start your preparation journey**
            
            Next Steps:
            1. Learn data structures basics
            2. Understand algorithm fundamentals
            3. Solve easy problems on LeetCode
            4. Watch tutorial series on YouTube
            5. Practice consistently every day
            """)
        
        st.markdown("---")
        
        # Assessment History
        if st.session_state.assessment_history:
            st.subheader("📅 Assessment History")
            
            history_data = []
            for assessment in st.session_state.assessment_history:
                avg = sum(assessment['scores'].values()) / len(assessment['scores'])
                history_data.append({
                    'Date': assessment['date'][:10],
                    'Average Score': f"{avg:.1f}%"
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

# ==================== SETTINGS ====================
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    st.subheader("🎨 Display Preferences")
    
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    
    st.subheader("🔔 Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("Email Notifications", value=True)
        daily_reminders = st.checkbox("Daily Study Reminders", value=True)
    
    with col2:
        progress_updates = st.checkbox("Weekly Progress Updates", value=True)
        question_suggestions = st.checkbox("Question Suggestions", value=True)
    
    st.markdown("---")
    st.subheader("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Data", use_container_width=True):
            export_data = {
                'profile': st.session_state.profile,
                'scores': st.session_state.scores,
                'assessment_history': st.session_state.assessment_history
            }
            st.json(export_data)
    
    with col2:
        if st.button("🗑️ Reset All Data", use_container_width=True):
            if st.checkbox("Are you sure? This cannot be undone."):
                st.session_state.profile = {}
                st.session_state.scores = {}
                st.session_state.assessment_history = []
                st.success("✅ Data reset successfully!")
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    
    st.markdown("""
    ### AI Interview Prep v2.0
    
    Your personal AI-powered coding interview coach.
    
    **Version:** 2.0.0
    
    **Built with:** Streamlit, Plotly, Scikit-learn
    
    **Last Updated:** July 2026
    
    **License:** MIT
    
    ### Contact & Support
    - 📧 Email: support@aiinterviewprep.com
    - 💬 Discord: Join our community
    - 🐛 Issues: Report on GitHub
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p>Made with ❤️ for interview aspirants | © 2026 AI Interview Prep</p>
    <p>
        <a href="#" style="color: #6366f1; text-decoration: none;">GitHub</a> • 
        <a href="#" style="color: #6366f1; text-decoration: none;">LinkedIn</a> • 
        <a href="#" style="color: #6366f1; text-decoration: none;">Twitter</a>
    </p>
</div>
""", unsafe_allow_html=True)