import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import random
import json

st.set_page_config(
    page_title="AI Interview Prep",
    layout="wide"
)

with open("question.json", "r") as file:
    question_bank = json.load(file)

st.sidebar.title("AI Interview Prep")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Student Profile",
        "Skill Assessment",
        "Dashboard",
        "AI Roadmap",
        "Interview Questions",
        "Progress Tracker"
    ]
)

if 'scores' not in st.session_state:
    st.session_state.scores = {}

if 'profile' not in st.session_state:
    st.session_state.profile = {}


if menu =="Home":
    st.title("AI Interview Prep")
    st.markdown("""
    ### Crack Technical Interviews with AI-Powered Analysis

    Analyze your coding skills, detect weak areas,
    generate interview questions, and build a smart roadmap.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Students Practicing", "100+")
    col2.metric("Questions Available", "500+")
    col3.metric("Success Rate", "87%")

    st.divider()

    st.subheader("🎯 What This System Can Do")

    col1, col2 = st.columns(2)

    with col1:
        st.info("📊 Skill Assessment")
        st.info("⚠ Weak Topic Detection")
        st.info("🛣 AI Learning Roadmap")

    with col2:
        st.info("💡 Interview Questions")
        st.info("📈 Progress Tracking")
        st.info("🤖 AI Readiness Prediction")

    st.divider()

    st.success("Start your preparation journey today")

elif menu == "Student Profile":

    st.title("🎓 Student Profile")

    name = st.text_input("Enter Your Name")

    role = st.selectbox(
        "Preferred Role",
        [
            "Frontend Developer",
            "Backend Developer",
            "Full Stack Developer",
            "Data Analyst",
            "ML Engineer",
            "AI Engineer",
            "Data Scientist"
        ]
    )

    branch = st.selectbox(
        "Branch",
        [
            "BCA",
            "BTech",
            "MCA",
            "BSc(IT)",
            "MSc(IT)"
        ]
    )

    year = st.selectbox(
        "Year",
        [
            "1st Year",
            "2nd Year",
            "3rd Year",
            "Final Year",
            "Passout"
        ]
    )

    if st.button("Save Profile"):

        st.session_state.profile = {
            "name": name,
            "role": role,
            "branch": branch,
            "year": year
        }

        st.success("Profile Saved Successfully!")

    if st.session_state.profile:

        st.subheader("Saved Profile")

        st.write("👤 Name:", st.session_state.profile["name"])
        st.write("💼 Role:", st.session_state.profile["role"])
        st.write("🎓 Branch:", st.session_state.profile["branch"])
        st.write("📅 Year:", st.session_state.profile["year"])
elif menu =="Skill Assessment":
    st.title("🔍 Skill Assessment")
    topics =[
        "Arrays",
        "linked List",
        "Stack",
        "Queue",
        "DBMS",
        "Operating Systems",
        "Programming Languages",
        "Data Structures",
        "Algorithms",
        "System Design",
        
    ]
    arrays = st.slider("Arrays",0,100,50)
    linked_list = st.slider("Linked List",0,100,50)
    stack = st.slider("Stack",0,100,50)
    queue = st.slider("Queue",0,100,50)
    dbms = st.slider("DBMS",0,100,50)
    os = st.slider("Operating Systems",0,100,50)
    pl = st.slider("Programming Languages",0,100,50)
    ds = st.slider("Data Structures",0,100,50)
    algorithms = st.slider("Algorithms",0,100,50)
    system_design = st.slider("System Design",0,100,50)
    
    if st.button("Save Scores"):

     st.session_state.scores = {
        "Arrays": arrays,
        "Linked List": linked_list,
        "Stack": stack,
        "Queue": queue,
        "DBMS": dbms,
        "Operating Systems": os,
        "Programming Languages": pl,
        "Data Structures": ds,
        "Algorithms": algorithms,
        "System Design": system_design
    }

    st.success("Scores Saved Successfully!")

    avg_score = (
        sum(st.session_state.scores.values())
        / len(st.session_state.scores)
    )

    st.subheader(f"Average Score: {avg_score:.2f}")

    if avg_score < 35:
        st.error(
            "Preparation level is LOW. Focus on fundamentals."
        )

    elif avg_score < 60:
        st.warning(
            "Preparation level is MEDIUM. Revise weak topics."
        )

    elif avg_score < 80:
        st.info(
            "Preparation level is GOOD. Keep practicing."
        )

    else:
        st.success(
            "Preparation level is EXCELLENT. Interview ready!"
        )

elif menu == "Dashboard":

    st.title("📊 Dashboard")

    if not st.session_state.profile:
        st.warning("Please complete Student Profile first")

    elif not st.session_state.scores:
        st.warning("Please complete Skill Assessment first")

    else:

        profile = st.session_state.profile
        scores = st.session_state.scores

        # Student Information
        st.subheader("👨‍🎓 Student Information")

        st.write("Name:", profile["name"])
        st.write("Role:", profile["role"])
        st.write("Branch:", profile["branch"])
        st.write("Year:", profile["year"])

        st.divider()

        # Metrics
        avg_score = sum(scores.values()) / len(scores)

        weak_topics = [
            topic for topic, score in scores.items()
            if score < 50
        ]

        strong_topics = [
            topic for topic, score in scores.items()
            if score >= 75
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Score",
            f"{avg_score:.2f}%"
        )

        col2.metric(
            "Weak Topics",
            len(weak_topics)
        )

        col3.metric(
            "Strong Topics",
            len(strong_topics)
        )

        st.divider()

        # Topic Table
        st.subheader("📚 Topic Performance")

        df = pd.DataFrame({
            "Topic": list(scores.keys()),
            "Score": list(scores.values())
        })

        st.dataframe(df)

        st.divider()

        # Weak Topics
        st.subheader("⚠ Weak Topics")

        if weak_topics:
            for topic in weak_topics:
                st.error(topic)
        else:
            st.success("No Weak Topics")

        # Strong Topics
        st.subheader("✅ Strong Topics")

        if strong_topics:
            for topic in strong_topics:
                st.success(topic)

        st.divider()

        # Feedback
        st.subheader("🤖 Feedback")

        if avg_score >= 80:
            st.success(
                "Excellent preparation level."
            )

        elif avg_score >= 60:
            st.info(
                "Good progress. Focus on weak topics."
            )

        else:
            st.warning(
                "You need more practice before interviews."
            )
elif menu == "AI Roadmap":

    st.title("AI Learning Roadmap")

    if not st.session_state.scores:
        st.warning("Please complete Skill Assessment first")

    else:
        scores = st.session_state.scores
        weak_topics = [k for k, v in scores.items() if v < 50]
        if not weak_topics:
            st.success(
                "Excellent Performance! Keep practicing advanced problems."
            )
        else:
            for topic in weak_topics:
                st.subheader(f"📚 {topic} Roadmap")
                if topic == "Arrays":
                    st.write("Week 1 :- Array Basics")
                    st.write("Week 2 :- Sliding Window")
                    st.write("Week 3 :- Prefix Sum")
                    st.write("Week 4 :- LeetCode Problems")

                elif topic == "Linked List":
                    st.write("Week 1 :- Linked List Basics")
                    st.write("Week 2 :- Reverse Linked List")
                    st.write("Week 3 :- Fast & Slow Pointer")
                    st.write("Week 4 :- Interview Problems")

                elif topic == "Stack":
                    st.write("Week 1 :- Stack Basics")
                    st.write("Week 2 :- Monotonic Stack")
                    st.write("Week 3 :- Expression Evaluation")
                    st.write("Week 4 :- Advanced Problems")

                elif topic == "Queue":
                    st.write("Week 1 :- Queue Basics")
                    st.write("Week 2 :- Circular Queue")
                    st.write("Week 3 :- Priority Queue")
                    st.write("Week 4 :- Graph Applications")

                elif topic == "DBMS":
                    st.write("Week 1 :- SQL Basics")
                    st.write("Week 2 :- Joins")
                    st.write("Week 3 :- Normalization")
                    st.write("Week 4 :- Transactions & Indexing")

                elif topic == "OS":
                    st.write("Week 1 :- Process & Thread")
                    st.write("Week 2 :- Scheduling")
                    st.write("Week 3 :- Deadlock")
                    st.write("Week 4 :- Memory Management")

elif menu=="Interview Questions":
    st.title("Interview Question Generator")

    topic=st.selectbox("Select Topic",
                       list(question_bank.keys()))
    
    difficulty=st.selectbox("Difficulty",
                            ["Easy","Medium","Hard"])
    
    if st.button("Generate Questions"):
        questions=random.sample(question_bank[topic],3)
        st.subheader(f"{difficulty} Level Questions")

        for i,q in enumerate(questions,start=1):
            st.write(f"{i}.{q}")

elif menu == "Progress Tracker":

    st.title(" Progress Tracker")

    if not st.session_state.scores:
        st.warning("Please complete Skill Assessment first")

    else:
        scores = st.session_state.scores
        progress_df = pd.DataFrame({
            'Topic': list(scores.keys()),
            'Current Score': list(scores.values()),
            'Target Score': [100] * len(scores)
        })

        st.dataframe(progress_df)

        st.subheader("Interview Readiness")
        avg_score = sum(scores.values()) / len(scores)

        X = [ [20], [35], [45], [55], [65], [75], [85], [95]]
        y = ['Low','Low','Medium','Medium','Good', 'Good','Excellent','Excellent' ]

        model = DecisionTreeClassifier()
        model.fit(X, y)
        prediction = model.predict([[avg_score]])

        st.success(f"Interview Readiness Level: {prediction[0]}")

        if prediction[0] == 'Low':
            st.error("You need strong preparation before placements")

        elif prediction[0] == 'Medium':
            st.warning("You are improving but still need practice")

        elif prediction[0] == 'Good':
            st.info("Good preparation level. Keep practicing")
        else:
            st.success("Excellent preparation level")