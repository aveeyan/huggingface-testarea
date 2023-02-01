# SCRIPT_NAME: huggingface_test_phrase.py
# 
# USAGE SYNTAX:
#   For IPython3: %run huggingface_test_phrase.py <path-to-input-file>
#   Example     : %run huggingface_test_phrase.py ./job-posting-sentences.txt
# 
# [BUG] UnboundError if the first line of the input file is a newline
#   

from mezcla.main import Main
from mezcla import debug
from mezcla import system
from transformers import pipeline
import random
import math

debug.set_level(3)

unmasker_wiki = pipeline("fill-mask", "albert-base-v2")
unmasker_jobs = pipeline("fill-mask", "./tmp/jobs-624298")

# [VARIABLE]: Test parameters
test_index = 0              # Test Indexing
case_index = 0              # Error Indexing
blank_index = 0             # Blank Sentences
single_index = 0            # Single Word Sentencs 
nonalpha_index = 0          # "Originally" selected character is not alphabet
albert_index = 0

dummy_main_app = Main(description=__doc__, skip_input=False, manual_input=True)
text = dummy_main_app.read_entire_input()


def get_score(model, token, sentence, verbose=False):
    """Get score for TOKEN in MODEL in masked SENTENCE"""
    # ex: get_score(model, "dog", "A [MASK] is man's best friend")
    try:
       result = model(sentence)
    except:
       system.print_exception_info("lookup")
       return 0
    
    score = 0
    found = False
    
    for entry in result:
        if (entry['token_str'] == token):
            score = entry['score']
            found = True
            break

    return score

def do_test():    
    
    global test_index, blank_index, single_index, nonalpha_index, case_index, albert_index

    test_details = []
    
    for sentence in text.splitlines():
        
        test_index += 1

        def single_word_detect(tokens):
            global single_index
            if len(tokens) == 1:
                single_index += 1
                print (f"""
        ==========
        [ALERT]: Sentence only contains one word.
        raw_word = {word_raw}
        single_index = {single_index}
        ==========""")


        def not_alpha_detect(raw_word):
            global nonalpha_index
            if not any(c.isalpha() for c in raw_word):
                nonalpha_index += 1
                print (f"""
        ==========
        [ALERT]: Randomly selected word is not an alphabet.
        word_raw: {raw_word}
        non_alpha = {nonalpha_index}
        ==========""")

        # [PROCESS]: Split into words and replace random one
        tokens = sentence.split()
        rand_pos = int(math.floor(len(tokens) * random.random())) 
        
        # # [OLD]: Using round may cause IndexError | IF is included to prevent such error
        # rand_pos = int(round(len(tokens) * random.random(), 0)) 
        # if rand_pos == len(tokens):
        #    floorerror += 1
        
        try:
            tokens_copy = tokens.copy()
            tokens_copy[rand_pos] = "[MASK]"
            masked_sentence = " ".join(tokens_copy)
            word_raw = tokens[rand_pos]
        except:
            case_index += 1
            system.print_exception_info(f"\n\t######## TEST [{test_index}] #########\tcase_index: {case_index}\n")
            test_index += 1
            pass
        
        # [PROCESS] Words filtered to lowercase as models accept lowercase only
        word = ''.join([char for char in word_raw if char.isalpha()]).lower()
        score_jobs = get_score(unmasker_jobs, word, masked_sentence)
        score_wiki = get_score(unmasker_wiki, word, masked_sentence)
        score_diff = score_jobs - score_wiki
        if (score_wiki == 0 and score_diff == 0):
            blank_index += 1

        # Applying sorting methods to sort tests
      
        albert_base_dominant =  False if score_diff >= 0 else True
        if albert_base_dominant: 
            albert_index += 1

        word_filter = False if word_raw == word else True
        pass_index = test_index - case_index
        # Appending details to test_details list if non_zero
        if (score_jobs != 0 and score_wiki != 0):
            testD = {}
            testD["test_index"] = test_index
            testD["word"] = word
            testD["word_filter"] = word_filter
            testD["sentence"] = masked_sentence
            testD["albert_score"] = score_wiki
            testD["jobs_score"] = score_jobs
            testD["albert_dominant"] = albert_base_dominant
            testD["score_diff"] = abs(score_diff)
            
            test_details.append(testD)

        OUTPUT_MODEL = f"""
    --------------------
    TEST [{test_index}]
    --------------------
        Word: {word_raw}
        Index: {rand_pos}
        word_filter: {word_filter}
        Sentence: {masked_sentence}
        
        Score [albert-base-v2]: {score_wiki}
        Score [jobs-642498]: {score_jobs}
        
        Score Difference: {abs(score_diff)}
        albert_base_dominant: {albert_base_dominant}

    UPDATED COUNT
    ==============
        TEST_ERROR: {case_index}
        SUCCESS: {pass_index}
            ATLEAST_ONE_SCORE: {pass_index - blank_index}
            NO_SCORE: {blank_index}
        ALBERT_TEST_DOMINANT: {albert_index}
        """
        
        print (OUTPUT_MODEL)
        not_alpha_detect(word_raw)
        single_word_detect(tokens)

    # Sorting Values from Test Details
    td_sort_albert = sorted(
        test_details,
        key=lambda x: x["albert_score"] 
        )
    
    td_sort_jobs = sorted(
        test_details,
        key=lambda x: x["jobs_score"] 
        )
    
    td_sort_diff = sorted(
        test_details,
        key=lambda x: x["score_diff"] 
        )

    # Top 5 Results Return

    def top5(td, score_str, is_desc):
        """Returns the top-5 results of the scores using the td_sort_AAA array"""
        start = 0
        stop = 5 if len(td) >= 5 else len(td)
        interval = 1
        test_index_str = "test_index"
        word_str = "word"
        albert_base_dominant = "albert_dominant"
        result = "S.N.\tScore\t\t\ttest_index\tword\t\t\talbert_dominant\n-----------------------------------------------------------------------------------------\n"
        if is_desc:
            start = -1
            stop = -6 if len(td) >= 5 else -len(td)-1
            interval *= -1
        # [TOFIX]: IndexError in the case of no items in td
        try:
            for i in range(start, stop, interval):
                result += str(f"{abs(i)}\t{round(td[i][score_str], 15)}\t{td[i][test_index_str]}\t\t{td[i][word_str]}\t\t\t{td[i][albert_base_dominant]}") + "\n"
        except Exception:
            result = None
    
        return result

    SUMMARY_OUTPUT_MODEL = f"""
===========================================
-------------------------------------------
END OF FILE: DETAILED SUMMARY
-------------------------------------------
[TEST_COUNT]
     TOTAL: {test_index}
    FAILED: {case_index}
    PASSED: {test_index - case_index}

[PASSED_TEST_SUMMARY]
    ATLEAST_ONE_SCORE: {pass_index - blank_index}
           BOTH_SCORE: {len(test_details)}
            BOTH_ZERO: {blank_index}

 SINGLE_WORD_SENTENCE: {single_index}
       NON_ALPHA_WORD: {nonalpha_index}

[BASE_DOMINANT]
    ALBERT_BASE: {albert_index}
 TMP/JOB_642498: {pass_index - len(test_details) - albert_index}

[MOST_SUCCESSFUL (NON_ZERO)]
    ALBERT_BASE: {td_sort_albert[-1]["albert_score"]}\t\t[test_index = {td_sort_albert[-1]["test_index"]} | word: {td_sort_albert[-1]["word"]}]
 TMP/JOB_642498: {td_sort_jobs[-1]["jobs_score"]}\t\t[test_index = {td_sort_jobs[-1]["test_index"]} | word: {td_sort_jobs[-1]["word"]}]

[LEAST_SUCCESSFUL (NON_ZERO)]
    ALBERT_BASE: {td_sort_albert[0]["albert_score"]}\t\t[test_index = {td_sort_albert[0]["test_index"]} | word: {td_sort_albert[0]["word"]}]
 TMP/JOB_642498: {td_sort_jobs[0]["jobs_score"]}\t\t[test_index = {td_sort_jobs[0]["test_index"]} | word: {td_sort_jobs[0]["word"]}]

[SCORE_DIFFERENCES (NON_ZERO)]
    HIGHEST_DIFF: {td_sort_diff[-1]["score_diff"]}\t\t[test_index = {td_sort_diff[-1]["test_index"]} | word: {td_sort_diff[-1]["word"]}]
     LOWEST_DIFF: {td_sort_diff[0]["score_diff"]}\t\t[test_index = {td_sort_jobs[0]["test_index"]} | word: {td_sort_jobs[0]["word"]}]

-------------------------------------------
LIST: TESTS_ON_THE_BASIS_OF_SCORES
-------------------------------------------

1. HIGHEST_SCORE (NON_ZERO) - TOP_5

[ALBERT_BASE]

{top5(td_sort_albert, "albert_score", True)}

[TMP/JOB_642498]

{top5(td_sort_jobs, "jobs_score", True)}


2. LOWEST_SCORE (NON_ZERO) - TOP_5

[ALBERT_BASE]

{top5(td_sort_albert, "albert_score", False)}

[TMP/JOB_642498]

{top5(td_sort_jobs, "jobs_score", False)}


3. HIGHEST_SCORE_DIFF (NON_ZERO) - TOP_5

{top5(td_sort_diff, "score_diff", True)}


4. LOWEST_SCORE_DIFF (NON_ZERO) - TOP_5

{top5(td_sort_diff, "score_diff", False)}

===========================================
"""
    print (SUMMARY_OUTPUT_MODEL)

    print(test_details)
    print(len(test_details))

def main():
    do_test()

if __name__ == "__main__":
    main()