import re


def makeDataSet(dataLines):
    dataSet = []

    for line in dataLines:
        state, tweet = line.split('\t', 1)
        label = getLabelStatePoliticalAffiliation(state)
        features = []

        unigrams = getUnigramsFilterOutBadTokens(tweet)
        bigrams = getBigrams(unigrams)



def getLabelStatePoliticalAffiliation(state):
    # if state is liberal, label = 0; if conservative, label = 1
    # declare lists of liberal states and conservative states
    liberalStates = ['WA', 'OR', 'CA', 'NV', 'CO', 'NM', 'IA', 'WI', 'MI', 'IL', 'OH', 'VA', 'DC', 'MD', 'DE', 'PA',
                     'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'VT', 'ME', 'HI', 'FL']

    conservativeStates = ['ID', 'MT', 'WY', 'UT', 'AZ', 'ND', 'SD', 'NE', 'KS', 'OK', 'TX', 'MO', 'AR', 'LA', 'IL',
                          'KY', 'TN', 'MS', 'AL', 'WV', 'NC', 'SC', 'GA', 'AK']

    if state in liberalStates:
        return 0
    elif state in conservativeStates:
        return 1


def getUnigramsFilterOutBadTokens(tweet):
    tokens = tweet.split()

    punctuation = ['.', ',', '?', '!', "'", '"', '(', ')', '&', '=', '/', '+', '-', '*']

    # filter out html, handles, and floating punctuation, RT
    tokens = [token for token in tokens if
              not token.startswith('http://') and not token.startswith('@') and not token.startswith(
                  'RT') and token not in punctuation]

    # remove emojis
    noemojis = []
    for token in tokens:
        if token.startswith('#'):
            tag = re.search('#([A-Za-z]|\d)+', token)
            if tag:
                noemojis.append(tag.group().lower())
        else:
            word = re.search(r'([A-Za-z]|\d)+(\'[A-Za-z]+)*', token)
            if word:
                noemojis.append(word.group().lower())

    unigrams = noemojis
    return unigrams


def getBigrams(unigrams):
    bigrams = []

    for i in xrange(len(unigrams) - 1):
        bigrams.append('%s %s' % (unigrams[i], unigrams[i + 1]))

    return bigrams


def getLiberalCount(unigrams, bigrams):
    # liberal lexicon
    liberalFoods = ['curry', 'bistro', 'fresh', 'fruit', 'strawberry', 'crunchy', 'thin', 'coconut', 'lamb', 'gnocchi',
                'fusili', 'radiatore', 'rice', 'wine', 'beer', 'diet', 'tap', 'fusion', 'vegetarian', 'foodie',
                'organic', 'seafood', 'toast', 'bagel', 'jamba', 'sbarro', 'chipotle', 'aubonpain', 'qdoba',
                'wienerschnitzel', 'starbucks', 'wingstop', 'panera', 'tacobell', 'quiznos', 'dunkin', 'donut',
                'dunkindonuts', 'pfchangs', 'cheesecakefactory', 'cpk', 'californiapizzakitchen', 'buffalowildwings',
                'ihop', 'wholefoods', 'whole foods', 'traderjoes', 'trader', "joe's", "trader joe's", 'safeway', 'frys',
                'fredmeyer', 'fred meyer', 'albertsons', 'osco', 'target', 'supertarget', 'macaroni', 'acme', 'blimpie',
                'ontheborder', 'asian']
    liberalFoodsHashtags = ['#' + l for l in liberalFoods]


def getConservativeCount(unigrams, bigrams):
    # conservative lexicon
    conservativeFoods = ['meatloaf', 'potato', 'bean', 'gravy', 'soda', 'mcdonalds', 'steak', 'cooked', 'grape', 'soft',
                'deep dish', 'burger', 'grill', 'tuna', 'casserole', 'meatloaf', 'linguine', 'rotini', 'spaghetti',
                'juice', 'chinese', 'cheeseburger', 'bacon', 'applebees', 'schlotzskys', 'chickfila', 'arbys', 'sonic',
                'checkers', 'hardees', 'dominos', 'mcdonalds', 'wendys', 'kfc', 'subway', 'panda', 'express',
                'pandaexpress', 'olivegarden', 'olive', 'garden', 'redlobster', 'goldencorral', 'buffet', 'hooters',
                'papamurphys', "murphy's", 'dennys', "denny's", 'krystal', 'dairy queen', 'dairyqueen', 'churchs',
                "church's", 'papajohns', "john's", 'krispykreme', 'krispy kreme', 'walmart', 'foodlion', 'food lion',
                'kroger', 'harristeeter', 'harris', 'teeter', 'publix']
    conservativeFoodsHashtags = ['#' + c for c in conservativeFoods]
