import numpy as np
from collections import defaultdict
from pprint import pprint
from operator import itemgetter

# 100 users watching 5 channels
X = np.zeros((100,5), dtype="bool")
features = ["Investing101","FunnyCatVideos","BuraTech","MInteractive","VideoGameReview"];

INVESTING101_INDEX = 0
FUNNYCATVIDEOS_INDEX = 1
BURATECH_INDEX = 2
MINTERACTIVE_INDEX = 3
VIDEOGAMEREVIEW_INDEX = 4

LIKED_VIDEO = 1

for viewer in range(X.shape[0]):
    if np.random.random() < 0.3:
        X[viewer][INVESTING101_INDEX] = LIKED_VIDEO
        if np.random.random() < 0.5:
            X[viewer][FUNNYCATVIDEOS_INDEX] = LIKED_VIDEO
        if np.random.random() < 0.2:
            X[viewer][BURATECH_INDEX] = LIKED_VIDEO
        if np.random.random() < 0.4:
            X[viewer][MINTERACTIVE_INDEX] = LIKED_VIDEO
        if np.random.random() < 0.6:
            X[viewer][VIDEOGAMEREVIEW_INDEX] = LIKED_VIDEO
    else:
        if np.random.random() < 0.5:
            X[viewer][FUNNYCATVIDEOS_INDEX] = LIKED_VIDEO

            if np.random.random() < 0.3:
                X[viewer][BURATECH_INDEX] = LIKED_VIDEO
            if np.random.random() < 0.2:
                X[viewer][MINTERACTIVE_INDEX] = LIKED_VIDEO
            if np.random.random() < 0.4:
                X[viewer][VIDEOGAMEREVIEW_INDEX] = LIKED_VIDEO
        else:
            if np.random.random() < 0.7:
                X[viewer][BURATECH_INDEX] = LIKED_VIDEO
                if np.random.random() < 0.55:
                    X[viewer][MINTERACTIVE_INDEX] = LIKED_VIDEO
                if np.random.random() < 0.8:
                    X[viewer][VIDEOGAMEREVIEW_INDEX] = LIKED_VIDEO

    if X[viewer].sum() == 0:
        X[viewer][VIDEOGAMEREVIEW_INDEX] = LIKED_VIDEO
    
# np.savetxt("viewer_profiles.txt", X, fmt="%d")
# np.loadtxt("viewer_profiles.txt")

# how many viewers lked the third channel ?
n_of_minteractive_likes = 0
for viewer in X:
    if viewer[MINTERACTIVE_INDEX] == 1:
        n_of_minteractive_likes += 1


# the hyptothesis :
rule_valid = 0
rule_invalid = 0

for viewer in X:
    if viewer[MINTERACTIVE_INDEX] == 1:
        if viewer[VIDEOGAMEREVIEW_INDEX] == 1:
            rule_valid += 1
        else:
            rule_invalid += 1
print("Check the hypothesis of : If a viewer like MInteractive, he/she will like VideoGameReview.")
print("{0} times the rule was valid.".format(rule_valid))
print("{0} times the rule was invalid.".format(rule_invalid))

support = rule_valid
confidence = rule_valid/n_of_minteractive_likes

print("{} Support.".format(support))
print("{} Confidence.".format(confidence))

# compute all the support and confidence to get the best correlation
n_of_samples, n_of_features = X.shape
n_of_occurences = defaultdict(int)
times_rule_valid = defaultdict(int)
times_rule_invalid = defaultdict(int)

for viewer in X:
    for premise in range(n_of_features):
        if viewer[premise] == 0: continue
        n_of_occurences[premise] += 1

        for conclusion in range(n_of_features):
            if premise == conclusion:
                continue
            if viewer[conclusion] == 1:
                times_rule_valid[(premise, conclusion)] += 1
            else:
                times_rule_invalid[(premise,conclusion)] += 1

support = times_rule_valid
confidence = defaultdict(float)

for premise, conclusion in times_rule_valid.keys():
    confidence[(premise,conclusion)] = times_rule_valid[(premise,conclusion)] / n_of_occurences[premise]

# if a viewer like the first channel he could like the second one
for premise, conclusion in confidence:
    first_channel = features[premise]
    second_channel = features[conclusion]
    # print("Rule : If a viewer likes {0}, he/she would like {1}.".format(first_channel, second_channel))
    # print("with a confidence of {}.".format(confidence[((premise,conclusion))]))
    # print("Support : {}.".format(support[((premise,conclusion))]))

# sort items 
support_sorted = sorted(support.items(),
                         key=itemgetter(1),
                           reverse=True)

def print_results(premise, conclusion, suport, confidence, features):
    first_channel = features[premise]
    second_channel = features[conclusion]

    print("Rule : If a viewer liked {0}, he/she would like {1}.".format(first_channel, second_channel))
    print("with a confidence of {}.".format(confidence[((premise,conclusion))]))
    print("Support : {}.".format(support[((premise,conclusion))]))

for index in range(5):
    (premise, conclusion) = support_sorted[index][0]
    print_results(premise, conclusion,support, confidence, features)

confidence_sorted = sorted(confidence.items(), 
                           key = itemgetter(1),
                           reverse=True)

print("----------------")
for index in range(5):
    (premise, conclusion) = confidence_sorted[index][0]
    print_results(premise, conclusion,support, confidence, features)