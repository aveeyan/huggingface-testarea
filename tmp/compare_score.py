from transformers import pipeline
from mezcla import debug
from mezcla import system

debug.set_level(3)

unmasker_wiki = pipeline('fill-mask', 'albert-base-v2')
unmasker_jobs = pipeline('fill-mask', './tmp/jobs-624298')

# Getting Scores for Jobs
def get_score_diff(token, sentence, verbose = True):

    """Get score difference for TOKEN in MODEL in masked SENTENCE"""
    try:
        result_jobs = unmasker_jobs(sentence)
        result_wiki = unmasker_wiki(sentence)
    except: 
        system.print_exception_info("lookup")
        return 0

    score_w = 0
    score_j = 0
    found = False
    dominant = None    # Wiki > Jobs: True else False

    for entry_j in result_jobs:
        if (entry_j['token_str'] == token):
            score_j = entry_j['score']
            found = True
            break

    for entry_w in result_wiki:
        if (entry_w['token_str'] == token):
            score_w = entry_w['score']
            found = True
            break
    
    score_diff = score_w - score_j
    
    if (verbose):

        score_j_round = round (score_j, 4)
        score_w_round = round (score_w, 4)
        
        if score_diff < 0:
            score_summary = f"jobs-624298 [{score_j_round}] > albert-base [{score_w_round}]"
            dominant = False
        elif score_diff > 0:
            score_summary = f"jobs-624298 [{score_j_round}] < albert-base [{score_w_round}]"
            dominant = True
        else:
            score_summary = "No result found in both models"
        
        print (f"\nalbert-base-dominant = {dominant}\nScore Summary: {score_summary}")
        print ("\n===============\nBest Results\n===============")
        print ("[albert-base]: " + ", ".join([v['token_str'] + ":" + system.round_as_str(v['score']) for v in result_wiki]))
        print ("[jobs-624298]: " + ", ".join([v['token_str'] + ":" + system.round_as_str(v['score']) for v in result_jobs]))
        print ("")

    return score_diff

        