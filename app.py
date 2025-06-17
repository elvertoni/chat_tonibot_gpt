import os
from decouple import config
import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils import (
    process_pdf,
    load_vector_store,
    add_to_vector_store,
    reset_vector_store
)

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

st.set_page_config(
    page_title='ToniBot RAG Agent',
    page_icon='🤖',
)
st.title('🤖 ToniBot — Agente IA com seus documentos')

knowledge_space = st.sidebar.text_input(
    '📚 Nome do espaço de conhecimento',
    value='default'
)
persist_directory = f'db/{knowledge_space}'

vector_store = load_vector_store(persist_directory)

with st.sidebar:
    st.subheader('🔗 Gerenciar Documentos')

    uploaded_files = st.file_uploader(
        '📄 Upload de PDFs',
        type=['pdf'],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner('📚 Processando documentos...'):
            all_chunks = []
            for file in uploaded_files:
                chunks = process_pdf(file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                persist_directory=persist_directory,
                vector_store=vector_store
            )
            st.success(f'✅ {len(all_chunks)} pedaços adicionados!')

    if st.button('🗑️ Resetar este espaço'):
        reset_vector_store(persist_directory)
        vector_store = None
        st.success('Espaço de conhecimento resetado!')

    st.sidebar.divider()

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
        'gpt-4.1-mini',
    ]
    selected_model = st.sidebar.selectbox(
        '🤖 Modelo LLM',
        model_options
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('💬 Pergunte algo...')

if not vector_store:
    st.warning('⚠️ Nenhum documento carregado. Faça upload na barra lateral.')

if vector_store and question:
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    llm = ChatOpenAI(model=selected_model)
    retriever = vector_store.as_retriever()

    system_prompt = """
    Você é um agente inteligente treinado nos documentos carregados.
    - Se não houver resposta no contexto, diga: "Não encontrei essa informação nos documentos."
    - Formate as respostas em **Markdown**.
    - Use listas, tabelas ou códigos quando fizer sentido.
    Contexto: {context}
    """

    messages = [('system', system_prompt)]
    for m in st.session_state.messages:
        messages.append((m['role'], m['content']))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=qa_chain,
    )

    with st.spinner('🤖 Gerando resposta...'):
        response = chain.invoke({'input': question}).get('answer')

    st.chat_message('ai').write(response)
    st.session_state.messages.append({'role': 'ai', 'content': response})
