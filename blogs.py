import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from Llama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    # Llama 2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    # Prompt Template
    template = """
        Write a blog for a {blog_style} job profile on the topic '{input_text}'
        within {no_words} words.
        """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    # Generate the response from the Llama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

# Streamlit app configuration
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

# Input fields
input_text = st.text_input("Enter the Blog Topic")

# Columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Displaying the result
if submit:
    if input_text and no_words.isdigit():
        with st.spinner('Generating your blog...'):
            response = getLLamaresponse(input_text, no_words, blog_style)
            st.subheader("Generated Blog:")
            st.write(response)
    else:
        st.error("Please provide a valid blog topic and a numeric value for the number of words.")
