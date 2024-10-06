import requests
import streamlit as st
from streamlit_chat import message
from chatbot_util import load_document, chunk_document, insert_or_fetch_embeddings, delete_pinecone_index
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import (HumanMessage, AIMessage)


doc = load_document()
chunks = chunk_document(doc)
vector_store_db = insert_or_fetch_embeddings("product", chunks)
print(vector_store_db)

retriever = vector_store_db.as_retriever(search_type='similarity', search_kwargs={'k': 3})

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

def ask_bot(question) :

    system_template = '''
    You are an assistant to interact on the products sales and customer experience analytics to help executives to understand the health / pulse of the current business of the organization.
    If you don't find the answer in the provided context, just respond "I don't know."
    '''

    user_template = f'''
    Question: ```{question}```
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    llm = AzureChatOpenAI(model="gpt-4o",openai_api_type="azure", temperature=1, azure_deployment="oaitrainingmodel-gpt-4o")
    ans = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type='stuff',
        combine_docs_chain_kwargs={'prompt': chat_prompt},
        verbose=True
    )
    return ans

def clear_history():
    if 'messages' in st.session_state:
        del st.session_state['messages']

if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Chatbot',
    )
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.subheader('Your AI Expert in Sales and Customer Experience Analytics.',
            help="Intelligent bot to explore sales trends, customer feedback, and performance metrics. Empower your decision-making with real-time analytics tailored to your business needs.")
    with st.sidebar:
        st.image('img.png', width=75, caption='InsightBot')
        clicked = st.button('Clear chat history!')
        if clicked:
            clear_history()

    question = st.chat_input( placeholder='Type your question here.')
    if question :

        st.session_state.messages.append(HumanMessage(content=question))
        with st.spinner('Generating response...'):
            response = ask_bot(question)
        st.session_state.messages.append(AIMessage(content=response.content))

        for i, msg in enumerate(st.session_state.messages):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=f'{i}')
            else:
                message(msg.content, is_user=False, key=f'{i}')

# url =  "https://cnadoaitraining.openai.azure.com/openai/deployments?api-version=2023-03-15-preview"
# response = requests.get(url, headers={"api-key": ""})
#
# if response.status_code == 200:
#     print(response.text)
# else:
#     print(f"Failed to retrieve data. Status code: {response.status_code}")
# delete_pinecone_index()