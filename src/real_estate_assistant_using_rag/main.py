import streamlit as st
from real_estate_assistant_using_rag.rag import process_url, generate_answer

st.title("Real Estate Research Tool")
 
st.sidebar.header("Article URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
 
# Separate slot just for status messages (not shared with the question box)
status = st.empty()
 
if st.sidebar.button("Process URLs"):
    urls = [u.strip() for u in (url1, url2, url3) if u.strip()]
    if not urls:
        status.warning("You must provide at least one valid URL")
    else:
        try:
            for message in process_url(urls):
                status.info(message)
            status.success("URLs processed. You can ask a question now.")
        except Exception as e:
            status.error(f"Could not process URLs: {e}")
 
query = st.text_input("Question")
if query:
    try:
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(query)
 
        st.header("Answer:")
        st.write(answer)
 
        source_lines = [s.strip() for s in sources.split("\n") if s.strip()]
        if source_lines:
            st.subheader("Sources:")
            for source in source_lines:
                st.write(source)
    except RuntimeError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")