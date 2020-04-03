import json

def write_yml(obj, prefix=""):
    out=""
    next_prefix = "  "+prefix
    if isinstance(obj,dict):
        for key in obj:
            next_obj = obj[key]
            if isinstance(next_obj, str):
                out+=prefix+"- "+key+": "+next_obj+"\n"
            else:
                out += prefix+key+":\n"
                out += write_yml(next_obj, prefix=next_prefix)
    elif isinstance(obj,list):
        for item in obj:
            if isinstance(item, str):
                out += prefix+"- "+item+"\n"
            else:
                out += write_yml(item,prefix=prefix)
    else:
        raise ValueError("Only list or dict allowed")
    return out

js = json.load(open("vera.json"))
qa = js['qa']

nlu = open("nlu.md", "w+")
stories = open("stories.md", "w+")
domain_dict = {"intents":[], "responses":[]}
for idq in qa:
    domain_dict["intents"].append("question_%s"%idq)

    questions = qa[idq]["questions"]
    answer = qa[idq]["answer"]

    domain_response_id="answer_%s"%idq
    domain_answer = {"text":answer}
    domain_dict['responses'].append({domain_response_id:domain_answer})

    nlu.write("## intent:question_%s\n"%idq)
    for q in questions:
        nlu.write("- %s\n"%q)
    nlu.write("\n")

    stories.write("## question_%s path\n"%idq)
    stories.write("* question_%s\n"%idq)
    stories.write("  -  answer_%s\n"%idq)
    stories.write("\n")

with open("../domain.yml", "w+") as domain:
    yaml = write_yml(domain_dict)
    domain.write(yaml)