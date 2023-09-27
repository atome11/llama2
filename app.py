import streamlit_authenticator as stauth
import streamlit as st
from streamlit_chat import message
from torch import cuda, bfloat16
import transformers
import yaml
from yaml.loader import SafeLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline


# sélection du modèle 
model_id = 'Llama-2-13b-chat-hf'
device = f'cuda:{cuda.current_device()}' #if cuda.is_available() else 'cpu'

# chargement du fichier d'identification
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# initialisation de l'écran de login
authenticator.login('Login', 'main')

# Fonctions du modèle
# --------------------------------------------------------------------------------------------------------------------------------
def initialize_model(model_id):
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        #config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        #use_auth_token=hf_auth
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model

def tokenizer(hf_auth,model_id):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    return tokenizer

def generate_text_pipeline(model):
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer('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','meta-llama/Llama-2-13b-chat-hf'),
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        #stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=4096,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm
    
def init_chain(llm,vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                                  memory=memory)
    return chain
    
# Fonctions du chatbot
# --------------------------------------------------------------------------------------------------------------------------------
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me about 8A", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))


if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title("8A Worker")
    
    # Init model 
    model = initialize_model(model_id)
    # Launch model & pipeline
    llm = generate_text_pipeline(model)
    # Init chain
    vector_store = load_local("~/vectorstore_0")
    chain = init_chain(llm,vector_store)
    
    # Initialize session state    
    initialize_session_state()
    # Display chat history
    display_chat_history()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
