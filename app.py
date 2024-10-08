import streamlit as st
from langchain_core.prompts import PromptTemplate
from streamlit_chat import message
from chatbot_util import  prepare_vectorstore
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (HumanMessage, AIMessage)

INDEX_NAME = "langchain-chatbot"
vector_store_db = prepare_vectorstore(INDEX_NAME)
retriever = vector_store_db.as_retriever()

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

llm = AzureChatOpenAI(model="gpt-4o", openai_api_type="azure", temperature=1,
                      azure_deployment="oaitrainingmodel-gpt-4o")

prompt_template = '''
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. respond the answer in english and keep the answer concise
---------------

Context: ```{context}```

Question: ```{question}```
Helpful Answer:
'''

chain_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    chain_type='stuff',
    combine_docs_chain_kwargs={'prompt': chain_prompt},
    verbose=True
)

def ask_bot(que) :
    ans = chain.invoke({"question" : que})
    return ans

def clear_history():
    if 'messages' in st.session_state:
        del st.session_state['messages']
    if 'memory' in st.session_state:
        del st.session_state['memory']

if __name__ == '__main__':
    st.set_page_config(
        page_title='AI chatbot',
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
        st.session_state.messages.append(AIMessage(content=response.get('answer')))

        for i, msg in enumerate(st.session_state.messages):
            if isinstance(msg, HumanMessage):
                message(msg.content, is_user=True, key=f'{i}')
            else:
                message(msg.content, is_user=False, key=f'{i}')