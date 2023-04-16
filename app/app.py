import os

import streamlit as st
from index import PERSIST_DIRECTORY, CalendarIndex
from preprocess import file_to_events

cal_index = CalendarIndex.load() if os.path.exists(PERSIST_DIRECTORY) else None

st.title("CalendarGPT")
tab_settings, tab_qna = st.tabs(["Setup", "Q&A"])

# Define settings
show_event_sample = False
show_source_events = True
with tab_settings:
    # Collect user info (to be used to prompt later)
    prompt_context = {
        "name": st.text_input("What's your name?"),
        "emails": st.text_input("What email(s) do you use?"),
    }

    # Collect calendar events
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload your calendar events as .ics files", accept_multiple_files=True
        )
        st.markdown(
            "Hint: you can [download your Google Calendar events as .ics files]"
            "(https://support.google.com/calendar/answer/37111?hl=en)."
        )
        submitted = st.form_submit_button("Submit")
        if submitted and uploaded_files:
            # Convert .ics to structured CalendarEvent objects
            events = []
            with st.spinner("Learning from your calendar..."):
                for file in uploaded_files:
                    events += file_to_events(file.getvalue())

                # Parse events into a dataframe
                if cal_index:
                    cal_index.add_events(events)
                else:
                    cal_index = CalendarIndex.from_events(events)
                cal_index.save()

    # Show the user what we've collected
    st.markdown("**What I know so far...**")
    name = prompt_context.get("name")
    emails = prompt_context.get("emails")
    st.write(f"Your name is {name}. " if name else "You haven't told me your name yet.")
    st.write(
        f"Your emails are {emails}."
        if emails
        else "You haven't told me your email yet."
    )
    num_events = (
        len(cal_index.df) if cal_index is not None and cal_index.df is not None else 0
    )
    st.write(f"You've shared `{num_events}` calendar events with me!")

    # What else to show on Q&A page - helpful for debugging!
    st.write("**Display Options**")
    show_event_sample = st.checkbox("Display a random sample of calendar events")
    show_source_events = st.checkbox("Show source events below query")

# Q&A Interface
with tab_qna:
    if cal_index is None:
        st.error("No index loaded! Please upload your calendar events first.")
        st.stop()

    # Show a random sample of events to help the user come up with a question
    if show_event_sample:
        df_sampled = cal_index.df.sample(25)
        if st.button("Resample"):
            df_sampled = cal_index.df.sample(25)
        st.dataframe(df_sampled)

    # User asks a question
    submitted = False
    with st.form("query-form"):
        query = st.text_input(
            label="Ask me anything about your calendar!",
            placeholder="When did I last go to Japan?",
        )
        submitted = st.form_submit_button("Submit")

    # Show response
    if submitted:
        if prompt_context:
            cal_index = CalendarIndex.load(prompt_context)
        with st.spinner("I'm thinking..."):
            response_text, df_source_events = cal_index.ask(query)
            st.markdown(response_text)

            if show_source_events:
                st.write("Source events:")
                st.dataframe(df_source_events)
