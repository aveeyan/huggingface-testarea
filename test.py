from transformers import pipeline
from mezcla import debug
from mezcla import system

debug.set_level(3)
unmasker_wiki = pipeline('fill-mask', 'albert-base-v2')
unmasker_jobs = pipeline('fill-mask', './tmp/jobs-624298')

def get_score(model, token, sentence, verbose=False):
    """Get score for TOKEN in MODEL in masked SENTENCE"""
    # ex: get_score(model, "dog", "A [MASK] is man's best friend")
    try:
       result = model(sentence)
    except:
       system.print_exception_info("lookup")
       return 0



    if verbose:
      print("All results: " + ", ".join([v['token_str'] + ":" + system.round_as_str(v['score']) for v in result]))
    score = 0

    found = False
    
    for entry in result:
        if (entry['token_str'] == token):
            score = entry['score']
            found = True
            break
    return score


