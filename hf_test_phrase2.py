# SCRIPT_NAME: huggingface_test_phrase2.py
# This script uses jobs-793224 model
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
unmasker_jobs = pipeline("fill-mask", "./tmp/job-793224")

# # For working purpose only, later to be shifted using Script()
# HIDE_SUMMARY = "hide"
# is_summary_hidden = True if HIDE_SUMMARY in system.get_args() else False

# [VARIABLE]: Test parameters
test_index = 0              # Test Indexing
case_index = 0              # Error Indexing
blank_index = 0             # Blank Sentences
single_index = 0            # Single Word Sentencs 
nonalpha_index = 0          # "Originally" selected character is not alphabet
albert_index = 0

HIDE_SUMMARY = "hide"

dummy_main_app = Main(
    description=__doc__, 
    skip_input=False,
    boolean_options=[(HIDE_SUMMARY, "hides summary stats, results simplified")]
    )

text = dummy_main_app.read_entire_input()

arg_array = system.get_args()
is_summary_hidden = True if f"--{HIDE_SUMMARY}" in arg_array else False

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

def do_test(hide_summary=is_summary_hidden):    
    
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

        OUTPUT_MODEL_SHORT = f"""
     Sentence#: {test_index}
          word: {word_raw}
        jscore: {score_jobs}
        wscore: {score_wiki}
    score_diff: {abs(score_diff)}
           J>W: {albert_base_dominant}
      sentence: {masked_sentence}
    """
        
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

    UPDATED COUNT [{test_index}]
    =================
        TEST_ERROR: {case_index}
        SUCCESS: {pass_index}
            ATLEAST_ONE_SCORE: {pass_index - blank_index}
            NO_SCORE: {blank_index}
        ALBERT_TEST_DOMINANT: {albert_index}
        """

        if not hide_summary:
            print (OUTPUT_MODEL)
        else:
            print (OUTPUT_MODEL_SHORT)

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

    def top5(td, is_desc):
        """Returns the top-5 results of the scores using the td_sort_AAA array"""
        # [TOFIX]: IndexError in the case of no items in td (not working)
        start = 0
        stop = 10 if len(td) >= 10 else len(td)
        interval = 1

        test_index_str = "test_index"
        word_str = "word"
        score_diff_str = "score_diff"
        wiki_score_str= "albert_score"
        jobs_score_str = "jobs_score"
        albert_base_dominant = "albert_dominant"
        sentence_str = "sentence"

        result = "Sent#\tWord\t\tJDiff\tJScore\tWScore\t(J<W)\tSentence\n-----------------------------------------------------------------------------------------\n"
        if is_desc:
            start = -1
            stop = -11 if len(td) >= 10 else -len(td)-1
            interval *= -1
        
        try:
            for i in range(start, stop, interval):
                result += str(f"{td[i][test_index_str]}\t{td[i][word_str]}\t\t{round(td[i][score_diff_str], 4)}\t{round(td[i][jobs_score_str], 4)}\t{round(td[i][wiki_score_str], 4)}\t{td[i][albert_base_dominant]}\t{td[i][sentence_str]}") + "\n"
            return result

        except Exception:
            return "None of the tests have both scores"
            pass

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

1. HIGHEST_SCORE (NON_ZERO) - TOP_10

[ALBERT_BASE]

{top5(td_sort_albert, True)}

[TMP/JOB_642498]

{top5(td_sort_jobs, True)}


2. LOWEST_SCORE (NON_ZERO) - TOP_10

[ALBERT_BASE]

{top5(td_sort_albert, False)}

[TMP/JOB_642498]

{top5(td_sort_jobs, False)}


3. HIGHEST_SCORE_DIFF (NON_ZERO) - TOP_10

{top5(td_sort_diff, True)}


4. LOWEST_SCORE_DIFF (NON_ZERO) - TOP_10

{top5(td_sort_diff, False)}

===========================================
"""

    SHORT_SUMMARY_OUTPUT_MODEL = f"""
--------------------------
End of File: Test Summary
--------------------------

    Total Tests = {test_index}
         Passed = {pass_index}
         Failed = {case_index}

            J<W = {albert_index}
"""
    if not hide_summary:
        print (SUMMARY_OUTPUT_MODEL)
    else:
        print (SHORT_SUMMARY_OUTPUT_MODEL)

    #print(test_details)
    #print(len(test_details))

def main():
    do_test()

if __name__ == "__main__":
    print (is_summary_hidden)
    main()