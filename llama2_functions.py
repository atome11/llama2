from torch import cuda, bfloat16
import transformers
import sys

model_id = 'meta-llama/Llama-13b-chat-hf'
device = f'cuda:{cuda.current_device()}' #if cuda.is_available() else 'cpu'


def initialize_model():
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

def tokenizer(hf_auth='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',model_id = 'meta-llama/Llama-2-13b-chat-hf')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    return tokenizer

def generate_text_pipeline(model):
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        #stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=4096,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    return generate_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "initialize":
            model = initialize_model()
        elif sys.argv[1] == "generate":
            model = initialize_model()
            generate_text = generate_text_pipeline(model)
    else:
        print("Please specify an argument: 'initialize' or 'generate'")
