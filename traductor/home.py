import torch
from transformers import pipeline, MarianMTModel,MarianTokenizer
from flask import Flask,Blueprint,jsonify,request



bp = Blueprint('home',__name__,url_prefix='/home')


# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",torch_dtype=torch.bfloat16, device_map="auto")


# message = [
#     {
#         "role":"system",
#         "content":"You are a developer experienced with scraping python"},

#     {   "role":"user",
#         "content":"makes a scraping model on https://dolarapi.com/docs/argentina/ and brings me in json format the different types of dollar purchase and sale values ​​that the api provides. Please order your answer from least to greatest taking into account the value of the dollar purchase."},
    
#     ]

# prompt = pipe.tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

# outputs = pipe(prompt, max_new_tokens=500,do_sample=True,temperature=0.2, top_k=50, top_p=0.95)

# print(outputs[0]["generated_text"])

def traductor(text):
    model_name = "Helsinki-NLP/opus-mt-es-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=True)

    inputs = tokenizer(text,return_tensors="pt")

    transforms = model.generate(
        **inputs
        # max_token = 50,
        # temperature = 0.8,
        # top_k = 50,
        # top_p = 0.95
    )

    traductor_text = tokenizer.decode(transforms[0],skip_special_tokens=True)
    return traductor_text

# text = "por donde tenemos que caminar?"
# ia = traductor(text)
# print(ia)



@bp.route("/traduc",  methods = ["POST"])
def index():
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            check = True        
        else:
            data = request.form
            check = False
        data_ = data.get('text')
        resp = traductor(data_)

        if check:
            json=jsonify({"message":resp}),201
            return json
        else:
            jsonify({"error":"algo fallo"}),400
    else:
        jsonify({"error":"INTERNAL"}),502




