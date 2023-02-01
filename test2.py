test_index = 0
SUMMARY_OUTPUT_MODEL = f"""
===========================================
-------------------------------------------
DETAILED SUMMARY: FIRST {test_index} TESTS
-------------------------------------------
[TEST_COUNT]
     TOTAL: {test_index}
    FAILED: failed
    PASSED: passed

[PASSED_TEST_SUMMARY]
    ATLEAST_ONE_SCORE: atleastone
            BOTH_ZERO: bothzero

 SINGLE_WORD_SENTENCE: single_word
       NON_ALPHA_WORD: nonalpha

[BASE_DOMINANT]
    ALBERT_BASE: albert_dominant
 TMP/JOB_642498: albert_dominant_comp

[MOST_SUCCESSFUL (NON_ZERO)]
    ALBERT_BASE: albert_num
 TMP/JOB_642498: tmp

[SCORE_DIFFERENCES (NON_ZERO)]
    HIGHEST_DIFF: highest
     LOWEST_DIFF: lowest

-------------------------------------------
LIST: TESTS_ON_THE_BASIS_OF_SCORES
-------------------------------------------

1. HIGHEST_SCORE (NON_ZERO) - TOP_5

    [ALBERT_BASE] = [IN LIST]
 [TMP/JOB_642498] = [IN LIST]


2. LOWEST_SCORE (NON_ZERO) - TOP_5

    [ALBERT_BASE] = [IN LIST]
 [TMP/JOB_642498] = [IN LIST]


3. HIGHEST_SCORE_DIFF (NON_ZERO) - TOP_5

    [ALBERT_BASE] = [IN LIST]
 [TMP/JOB_642498] = [IN LIST]


4. LOWEST_SCORE_DIFF (NON_ZERO) - TOP_5

    [ALBERT_BASE] = [IN LIST]
 [TMP/JOB_642498] = [IN LIST]

===========================================
"""

print(SUMMARY_OUTPUT_MODEL)

test_details = []
# test_details["test_index"] = 1
# test_details["word"] = "hello_world"
# test_details["sentence"] = "The hello_world is your first piece of code"
# test_details["albert_score"] = 0.54
# test_details["jobs_score"] = 0.66
# test_details["albert_dominant"] = True if test_details["albert_score"] > test_details["jobs_score"] else False
# test_details["score_diff"] = abs(test_details["albert_score"] - test_details["jobs_score"])

for i in range(10):
    testD = {}
    testD["test_index"] = i
    testD["word"] = "hello_world"
    testD["sentence"] = "The hello_world is your first piece of code"
    testD["albert_score"] = 0.54
    testD["jobs_score"] = 0.66
    testD["albert_dominant"] = True if testD["albert_score"] > testD["jobs_score"] else False
    testD["score_diff"] = abs(testD["albert_score"] - testD["jobs_score"])
    test_details.append(testD)


print (test_details)

def type1():
    for i in range(10, 0, -1):
        print (i)

type1()