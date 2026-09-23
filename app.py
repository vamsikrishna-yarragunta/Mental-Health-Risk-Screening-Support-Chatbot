import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import pickle
import os
from dotenv import load_dotenv
from groq import Groq
from rag_utils import retrieve_relevant_context

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(
    page_title="Mental Health Risk Screening Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_data
def load_data():

    engine = create_engine(f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
    df = pd.read_sql("SELECT * FROM survey_data", con=engine)
    return df


df = load_data()


@st.cache_resource
def load_model():

    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    return rf_model, label_encoders


rf_model, label_encoders = load_model()


if "page" not in st.session_state:
    st.session_state.page = "📊 Dashboard"


if "support_chat_history" not in st.session_state:

    st.session_state.support_chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I'm here to provide general mental health information, wellness guidance, and supportive resources. How can I help you today?"
        }
    ]


if "risk_level" not in st.session_state:
    st.session_state.risk_level = None


if "risk_percentage" not in st.session_state:
    st.session_state.risk_percentage = None


# Handle floating chatbot button

if st.query_params.get("chatbot_open") == "1":

    st.session_state.page = "💬 Chatbot"

    st.query_params.clear()

    st.rerun()


# ================= CUSTOM CSS =================

st.markdown(
    """
    <style>

    /* Main Header */

    .app-header {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .app-header h1 {
        font-size: 38px;
        margin-bottom: 5px;
        font-weight: 700;
    }

    .app-header p {
        font-size: 16px;
        color: #6b7280;
        margin-top: 0;
    }


    /* Navigation */

    .navigation-label {
        text-align: center;
        font-size: 15px;
        color: #6b7280;
        margin-bottom: 8px;
        font-weight: 500;
    }


    div.stButton > button {
        height: 72px;
        border-radius: 14px;
        border: 1px solid #d1d5db;
        transition: all 0.2s ease;
        padding: 5px 15px;
    }


    div.stButton > button p {
        font-size: 25px !important;
        font-weight: 700 !important;
    }


    div.stButton > button:hover {
        border-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }


    /* Filters */

    div[data-testid="stExpander"] summary p {
        font-size: 22px !important;
        font-weight: 700 !important;
    }


    div[data-testid="stExpander"] label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }


    div[data-baseweb="select"] {
        font-size: 16px;
    }


    div[data-testid="stSlider"] {
        padding-top: 5px;
    }


    /* Floating Chatbot */

    .floating-chatbot {
        position: fixed;
        bottom: 30px;
        right: 35px;
        width: 68px;
        height: 68px;
        border-radius: 50%;
        background: #2563eb;
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
        font-size: 31px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.25);
        z-index: 999999;
        transition: all 0.2s ease;
    }


    .floating-chatbot:hover {
        transform: scale(1.08);
        background: #1d4ed8;
    }


    .chat-tooltip {
        position: fixed;
        bottom: 108px;
        right: 25px;
        background: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13px;
        display: none;
        z-index: 999999;
    }


    .floating-chatbot:hover + .chat-tooltip {
        display: block;
    }


    /* Chat Messages */

    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 12px 0;
    }


    .user-bubble {
        background-color: #e8f0fe;
        padding: 12px 18px;
        border-radius: 15px 15px 3px 15px;
        max-width: 70%;
        color: #1f2937;
        font-size: 16px;
    }


    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 12px 0;
    }


    .assistant-bubble {
        background-color: #f3f4f6;
        padding: 14px 18px;
        border-radius: 15px 15px 15px 3px;
        max-width: 75%;
        color: #1f2937;
        font-size: 16px;
        line-height: 1.6;
    }


    .chat-label {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 5px;
        color: #6b7280;
    }


    /* Chatbot Title */

    .chatbot-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }


    .chatbot-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================= HEADER =================

st.markdown(
    """
    <div class="app-header">
        <h1>🧠 Mental Health Risk Screening</h1>
        <p>Mental health insights, screening and supportive guidance</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ================= NAVIGATION =================

st.markdown(
    '<div class="navigation-label">Application Navigation</div>',
    unsafe_allow_html=True
)


nav1, nav2, nav3 = st.columns(3, gap="medium")


with nav1:

    if st.button(
        "📊  Dashboard",
        use_container_width=True,
        type="primary"
        if st.session_state.page == "📊 Dashboard"
        else "secondary"
    ):

        st.session_state.page = "📊 Dashboard"

        st.rerun()


with nav2:

    if st.button(
        "📝  Risk Assessment",
        use_container_width=True,
        type="primary"
        if st.session_state.page == "📝 Risk Assessment"
        else "secondary"
    ):

        st.session_state.page = "📝 Risk Assessment"

        st.rerun()


with nav3:

    if st.button(
        "💬  Support Chatbot",
        use_container_width=True,
        type="primary"
        if st.session_state.page == "💬 Chatbot"
        else "secondary"
    ):

        st.session_state.page = "💬 Chatbot"

        st.rerun()


st.divider()


page = st.session_state.page


# ============================================================
# ================= DASHBOARD PAGE ===========================
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        """
        This dashboard explores patterns in workplace mental health survey data —
        who seeks treatment, what factors correlate with it, and where the gaps in
        support and awareness might be.
        """
    )


    with st.expander("🔎 Filters", expanded=False):

        filter1, filter2, filter3, filter4 = st.columns(4)


        with filter1:

            gender_filter = st.multiselect(
                "Gender",
                options=df["gender"].unique(),
                default=df["gender"].unique()
            )


        with filter2:

            age_min = int(df["age"].min())

            age_max = int(df["age"].max())

            age_filter = st.slider(
                "Age Range",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max)
            )


        with filter3:

            family_history_filter = st.multiselect(
                "Family History",
                options=df["family_history"].unique(),
                default=df["family_history"].unique()
            )


        with filter4:

            top_countries = df["country"].value_counts().head(10).index

            df["country_grouped"] = df["country"].where(
                df["country"].isin(top_countries),
                "Other"
            )

            country_filter = st.multiselect(
                "Country",
                options=df["country_grouped"].unique(),
                default=df["country_grouped"].unique()
            )


    filtered_df = df[
        (df["gender"].isin(gender_filter)) &
        (df["age"] >= age_filter[0]) &
        (df["age"] <= age_filter[1]) &
        (df["family_history"].isin(family_history_filter)) &
        (df["country_grouped"].isin(country_filter))
    ]


    col1, col2, col3, col4 = st.columns(4)


    col1.metric("Total Respondents", len(filtered_df))

    col2.metric("Sought Treatment", f"{(filtered_df['treatment'] == 'Yes').mean() * 100:.1f}%")

    col3.metric("Family History", f"{(filtered_df['family_history'] == 'Yes').mean() * 100:.1f}%")

    col4.metric("Avg Age", f"{filtered_df['age'].mean():.0f}")
    

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Risk Factors",
            "Workplace Support",
            "Correlations"
        ]
    )


    # ================= OVERVIEW =================

    with tab1:

        col1, col2 = st.columns(2)


        with col1:

            fig = px.bar(
                filtered_df["gender"].value_counts().reset_index(),
                x="gender",
                y="count",
                title="Gender Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            fig = px.histogram(
                filtered_df,
                x="age",
                nbins=20,
                title="Age Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        fig = px.bar(
            filtered_df["treatment"].value_counts().reset_index(),
            x="treatment",
            y="count",
            title="Treatment Seeking Overall",
            color="treatment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ================= RISK FACTORS =================

    with tab2:

        col1, col2 = st.columns(2)


        with col1:

            fam_treat = pd.crosstab(
                filtered_df["family_history"],
                filtered_df["treatment"]
            )

            fig = px.bar(
                fam_treat,
                barmode="group",
                title="Treatment by Family History"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            work_treat = pd.crosstab(
                filtered_df["work_interfere"],
                filtered_df["treatment"]
            )

            fig = px.bar(
                work_treat,
                barmode="group",
                title="Treatment by Work Interference"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        company_treat = (
            filtered_df.groupby("no_employees")["treatment"]
            .value_counts(normalize=True)
            .unstack()
        )


        fig = px.bar(
            company_treat,
            barmode="stack",
            title="Treatment Rate by Company Size"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ================= WORKPLACE SUPPORT =================

    with tab3:

        col1, col2 = st.columns(2)


        with col1:

            benefits_treat = pd.crosstab(
                filtered_df["benefits"],
                filtered_df["treatment"]
            )


            fig = px.bar(
                benefits_treat,
                barmode="group",
                title="Treatment by Benefits Offered"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            care_treat = pd.crosstab(
                filtered_df["care_options"],
                filtered_df["treatment"]
            )


            fig = px.bar(
                care_treat,
                barmode="group",
                title="Treatment by Care Options Awareness"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        comfort_df = pd.DataFrame(
            {
                "Supervisor": filtered_df["supervisor"].value_counts(),
                "Coworkers": filtered_df["coworkers"].value_counts()
            }
        )


        fig = px.bar(
            comfort_df,
            barmode="group",
            title="Comfort Discussing MH: Supervisor vs Coworkers"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ================= CORRELATIONS =================

    with tab4:

        cols_to_encode = [
            "treatment",
            "family_history",
            "work_interfere",
            "remote_work",
            "benefits",
            "care_options"
        ]


        corr_df = filtered_df[cols_to_encode].apply(
            lambda x: pd.factorize(x)[0]
        )


        corr_matrix = corr_df.corr()


        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.info(
            "Note: Correlation direction (sign) depends on category encoding "
            "order and isn't inherently meaningful — only the strength "
            "(magnitude) matters here."
        )


    # ================= KEY INSIGHTS =================

    st.divider()

    st.header("📌 Key Insights")


    with st.expander("Click to see full analysis notes"):

        st.markdown(
            """
            - **Family history** is the strongest predictor of treatment-seeking (~77% vs ~34%)
            - **Work interference** is the second strongest — higher interference means much higher treatment-seeking
            - **Awareness of care options** correlates with higher treatment-seeking
            - **Benefits offered** has a weak but positive relationship
            - **Company size** and **remote work** show almost no effect
            - **Stigma** is more about ambiguity ("maybe") than outright fear
            """
        )


    with st.expander("View Raw Data"):

        st.dataframe(filtered_df)


    # ================= FLOATING CHATBOT =================

    st.markdown(
        """
        <a class="floating-chatbot"
           href="?chatbot_open=1"
           target="_self">
           💬
        </a>

        <div class="chat-tooltip">
            Open Support Assistant
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ================= RISK ASSESSMENT PAGE =====================
# ============================================================

elif page == "📝 Risk Assessment":

    st.title("Treatment-Seeking Risk Assessment")


    st.markdown(
        "Enter the information below to get an estimated treatment-seeking "
        "probability based on our machine learning model."
    )


    st.info(
        "Please complete all fields before calculating the estimated risk."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        age_input = st.number_input(
            "Age",
            min_value=18,
            max_value=72,
            value=None,
            placeholder="Enter your age",
            step=1
        )


        gender_input = st.selectbox(
            "Gender",
            label_encoders["gender"].classes_,
            index=None,
            placeholder="Select your gender"
        )


        family_history_input = st.selectbox(
            "Family History of Mental Illness",
            label_encoders["family_history"].classes_,
            index=None,
            placeholder="Select an option"
        )


        work_interfere_input = st.selectbox(
            "Does it interfere with your work?",
            label_encoders["work_interfere"].classes_,
            index=None,
            placeholder="Select an option",
            help="Select 'Not Applicable' if this doesn't apply to you"
        )


        no_employees_input = st.selectbox(
            "Company Size",
            label_encoders["no_employees"].classes_,
            index=None,
            placeholder="Select an option"
        )


    with col2:

        remote_work_input = st.selectbox(
            "Remote Work?",
            label_encoders["remote_work"].classes_,
            index=None,
            placeholder="Select an option"
        )


        tech_company_input = st.selectbox(
            "Tech Company?",
            label_encoders["tech_company"].classes_,
            index=None,
            placeholder="Select an option"
        )


        benefits_input = st.selectbox(
            "Mental Health Benefits Offered?",
            label_encoders["benefits"].classes_,
            index=None,
            placeholder="Select an option"
        )


        care_options_input = st.selectbox(
            "Aware of Care Options?",
            label_encoders["care_options"].classes_,
            index=None,
            placeholder="Select an option",
            help="Are you aware of the mental health care resources or options provided by your employer?"
        )


        wellness_program_input = st.selectbox(
            "Wellness Program Offered?",
            label_encoders["wellness_program"].classes_,
            index=None,
            placeholder="Select an option",
            help="A wellness program refers to employer-sponsored initiatives supporting overall health, such as counseling, fitness, or stress-management resources"
        )


    with col3:

        seek_help_input = st.selectbox(
            "Employer Encourages Seeking Help?",
            label_encoders["seek_help"].classes_,
            index=None,
            placeholder="Select an option"
        )


        anonymity_input = st.selectbox(
            "Anonymity Protected?",
            label_encoders["anonymity"].classes_,
            index=None,
            placeholder="Select an option",
            help="Is your identity protected if you choose to use mental health or substance abuse resources through your employer?"
        )


        leave_input = st.selectbox(
            "Ease of Taking Mental Health Leave",
            label_encoders["leave_"].classes_,
            index=None,
            placeholder="Select an option"
        )


        mh_consequence_input = st.selectbox(
            "Fear of Mental Health Consequences?",
            label_encoders["mental_health_consequence"].classes_,
            index=None,
            placeholder="Select an option"
        )


        phys_consequence_input = st.selectbox(
            "Fear of Physical Health Consequences?",
            label_encoders["phys_health_consequence"].classes_,
            index=None,
            placeholder="Select an option"
        )


    if st.button("Calculate Risk", type="primary"):

        required_inputs = {
            "Age": age_input,
            "Gender": gender_input,
            "Family History": family_history_input,
            "Work Interference": work_interfere_input,
            "Company Size": no_employees_input,
            "Remote Work": remote_work_input,
            "Tech Company": tech_company_input,
            "Benefits": benefits_input,
            "Care Options": care_options_input,
            "Wellness Program": wellness_program_input,
            "Seek Help": seek_help_input,
            "Anonymity": anonymity_input,
            "Leave": leave_input,
            "Mental Health Consequence": mh_consequence_input,
            "Physical Health Consequence": phys_consequence_input
        }


        missing_fields = [
            name
            for name, value in required_inputs.items()
            if value is None
        ]


        if missing_fields:

            st.warning(
                "Please complete all fields before calculating the risk."
            )


            st.write(
                "Missing:",
                ", ".join(missing_fields)
            )


            st.stop()


        user_input = pd.DataFrame(
            [
                {
                    "age": age_input,
                    "gender": gender_input,
                    "family_history": family_history_input,
                    "work_interfere": work_interfere_input,
                    "no_employees": no_employees_input,
                    "remote_work": remote_work_input,
                    "tech_company": tech_company_input,
                    "benefits": benefits_input,
                    "care_options": care_options_input,
                    "wellness_program": wellness_program_input,
                    "seek_help": seek_help_input,
                    "anonymity": anonymity_input,
                    "leave_": leave_input,
                    "mental_health_consequence": mh_consequence_input,
                    "phys_health_consequence": phys_consequence_input
                }
            ]
        )


        for col in user_input.columns:

            if col in label_encoders:

                user_input[col] = label_encoders[col].transform(
                    user_input[col]
                )


        risk_percentage = (
            rf_model.predict_proba(user_input)[0][1] * 100
        )


        if risk_percentage >= 70:

            risk_level = "High"
            color = "red"

        elif risk_percentage >= 40:

            risk_level = "Moderate"
            color = "orange"

        else:

            risk_level = "Low"
            color = "green"


        st.session_state.risk_level = risk_level

        st.session_state.risk_percentage = risk_percentage


        st.divider()


        st.subheader("📈 Assessment Result")


        st.markdown(
            f"### Estimated Treatment-Seeking Probability: "
            f"<span style='color:{color}; font-size:32px; "
            f"font-weight:bold;'>{risk_percentage:.1f}%</span>",
            unsafe_allow_html=True
        )


        st.progress(int(risk_percentage))


        st.markdown(
            f"### Risk Level: "
            f"<span style='color:{color}; font-weight:bold; "
            f"font-size:24px;'>{risk_level}</span>",
            unsafe_allow_html=True
        )


        if risk_level == "High":

            st.error(
                "The model estimates a relatively high probability of "
                "treatment-seeking based on the provided survey patterns."
            )


        elif risk_level == "Moderate":

            st.warning(
                "The model estimates a moderate probability of "
                "treatment-seeking based on the provided survey patterns."
            )


        else:

            st.success(
                "The model estimates a relatively low probability of "
                "treatment-seeking based on the provided survey patterns."
            )


        st.caption(
            "This is a statistical estimate based on survey patterns, "
            "not a clinical diagnosis. If you're struggling, please "
            "consider speaking with a mental health professional."
        )


# ============================================================
# ================= CHATBOT PAGE =============================
# ============================================================

elif page == "💬 Chatbot":

    back_col, title_col = st.columns([1, 8])


    with back_col:

        if st.button("← Back"):

            st.session_state.page = "📊 Dashboard"

            st.rerun()


    with title_col:

        st.markdown(
            '<div class="chatbot-title">💬 Mental Health Support Assistant</div>',
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="chatbot-subtitle">'
        'A supportive assistant for general mental health information '
        'and wellness guidance.'
        '</div>',
        unsafe_allow_html=True
    )


    for msg in st.session_state.support_chat_history:

        if msg["role"] == "user":

            st.markdown(
                f"""
                <div class="user-message">
                    <div class="user-bubble">
                        <div class="chat-label">You</div>
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                f"""
                <div class="assistant-message">
                    <div class="assistant-bubble">
                        <div class="chat-label">Support Assistant</div>
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    user_message = st.chat_input(
        "Ask a question about stress, anxiety, sleep, burnout, or mental wellness..."
    )


    if user_message:

        st.session_state.support_chat_history.append(
            {
                "role": "user",
                "content": user_message
            }
        )


        crisis_keywords = [
            "kill myself",
            "suicide",
            "suicidal",
            "end my life",
            "take my life",
            "hurt myself",
            "harm myself",
            "self harm",
            "self-harm"
        ]


        is_crisis = any(
            keyword in user_message.lower()
            for keyword in crisis_keywords
        )


        if is_crisis:

            bot_reply = """
If you feel that you may hurt yourself or are in immediate danger, please seek immediate help from emergency services or go to the nearest emergency department.

In the U.S., you can call or text **988** to reach the Suicide & Crisis Lifeline.

If possible, stay with someone you trust and let them know that you need support.

You do not have to handle this alone.
"""


        else:

            # Retrieve relevant knowledge base content for this specific question
            retrieved_context = retrieve_relevant_context(user_message)

            assessment_context = ""


            if (
                st.session_state.risk_level is not None
                and st.session_state.risk_percentage is not None
            ):

                assessment_context = f"""
The user has already completed the application's separate Risk Assessment.

Existing result:
Risk Level: {st.session_state.risk_level}
Estimated Treatment-Seeking Probability: {st.session_state.risk_percentage:.1f}%

This is a statistical screening result and NOT a medical diagnosis.

Do not perform another assessment.
Do not calculate a new risk score.

If the user asks about this result, explain the existing result in simple language.
"""


            system_prompt = f"""
You are a Mental Health Support Assistant.

Your purpose is to provide general mental health information,
wellness guidance, coping strategies, and supportive resources.

You are NOT a doctor or therapist.

Do not diagnose mental health conditions.
Do not prescribe medication.
Do not provide medical treatment.
Do not perform a new risk assessment.
Do not ask the user the questions from the application's Risk Assessment module.

The application already has a separate Risk Assessment section.
The chatbot has a different purpose.

You should have a natural conversation with the user and help with topics such as:

- Stress management
- Anxiety
- Burnout
- Sleep
- Work-life balance
- Relaxation techniques
- Healthy routines
- Emotional well-being
- General mental health information
- Finding appropriate professional support

Be empathetic, respectful, professional, and conversational.

If the user asks something unrelated to mental health, wellness, stress, 
work-life balance, or the app's own risk assessment results (for example: 
coding help, recipes, homework, general trivia, or other unrelated topics), 
politely decline and redirect them back to mental health and wellness 
topics. Do not answer unrelated questions, even if you know the answer.

Example redirect: "I'm here specifically to help with mental health and 
wellness topics. Is there something on your mind related to stress, sleep, 
work, or emotional wellbeing I can help with?"

When the user asks for a specific number of steps, points, or items, provide the complete requested number.
Do not stop early.
Keep each point reasonably concise so the complete answer can be provided.

Give clear and useful answers without unnecessarily repeating disclaimers.

If the user asks about their existing assessment result,
explain the existing result without treating it as a diagnosis.

Use the following reference information to help answer the user's question
accurately, if relevant. Do not mention that this information was "retrieved"
or reference a "knowledge base" explicitly — just use it naturally to inform
your response:

{retrieved_context}

{assessment_context}
"""


            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]


            for msg in st.session_state.support_chat_history[-10:]:

                messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )


            with st.spinner("Preparing a response..."):

                response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=2500
                )


                bot_reply = response.choices[0].message.content


        st.session_state.support_chat_history.append(
            {
                "role": "assistant",
                "content": bot_reply
            }
        )


        st.rerun()