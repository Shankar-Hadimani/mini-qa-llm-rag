import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify

# Set the cache directory to a new location with sufficient space
os.environ['HF_HOME'] = '/run/cache/'

app = Flask(__name__)

# Load the Hugging Face token from the environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Load the tokenizer and model with the token
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", use_auth_token=hf_token)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 0.0)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, temperature=temperature)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11434)
