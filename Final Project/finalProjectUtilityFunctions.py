import re


def makeDataSet(dataLines):
    dataSet = []

    for line in dataLines:
        state, tweet = line.split('\t', 1)
        label = getLabelStatePoliticalAffiliation(state)
        features = []

        unigrams = getUnigramsFilterOutBadTokens(tweet)
        bigrams = getBigrams(unigrams)

        liberalCount = getLiberalCount(unigrams, bigrams)
        conservativeCount = getConservativeCount(unigrams, bigrams)

        if liberalCount > 0:
            features.append("Liberal Count: %i" % liberalCount)
        if conservativeCount > 0:
            features.append("Conservative Count: %i" % conservativeCount)

        for unigram in unigrams:
            features.append(unigram)
        for bigram in bigrams:
            features.append(bigram)

        tweetinfo = {'state': state, 'label': label, 'features': features}
        dataSet.append(tweetinfo)

    return dataSet




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

    # filter out stopwords
    stopWordsList = getStopWordsList()
    tokens = [token for token in tokens if token not in stopWordsList]

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

    liberalCount = 0
    for unigram in unigrams:
        if unigram in liberalFoods:
            liberalCount += 1
        if unigram in liberalFoodsHashtags:
            liberalCount += 1

    for bigram in bigrams:
        if bigram in liberalFoods:
            liberalCount += 1
        if bigram in liberalFoodsHashtags:
            liberalCount += 1

    return liberalCount


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

    conservativeCount = 0
    for unigram in unigrams:
        if unigram in conservativeFoods:
            conservativeCount += 1
        if unigram in conservativeFoodsHashtags:
            conservativeCount += 1

    for bigram in bigrams:
        if bigram in conservativeFoods:
            conservativeCount += 1
        if bigram in conservativeFoodsHashtags:
            conservativeCount += 1

    return conservativeCount


def getStopWordsList():
    # return a list of stopwords
    stopwordsList = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone',
                     'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any',
                     'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask',
                     'asked', 'asking', 'asks', 'at', 'away', 'back', 'backed', 'backing', 'backs', 'be', 'became',
                     'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best',
                     'better', 'between', 'big', 'both', 'but', 'by', 'came', 'can', 'cannot', 'case', 'cases',
                     'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'did', 'differ', 'different',
                     'differently', 'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'each',
                     'early', 'either', 'end', 'ended', 'ends', 'enough', 'even', 'evenly', 'ever', 'every',
                     'everybody', 'everyone', 'everything', 'everywhere', 'face', 'faces', 'fact', 'facts', 'far',
                     'felt', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further',
                     'furthered', 'furthering', 'furthers', 'gave', 'general', 'generally', 'get', 'gets', 'give',
                     'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest',
                     'group', 'grouped', 'grouping', 'groups', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                     'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if',
                     'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it',
                     'its', 'itself', 'just', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'large',
                     'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long',
                     'longer', 'longest', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'men', 'might',
                     'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'necessary', 'need',
                     'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non',
                     'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'of', 'off', 'often', 'old',
                     'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or',
                     'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'part',
                     'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed',
                     'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems',
                     'put', 'puts', 'quite', 'rather', 'really', 'right', 'room', 'rooms', 'said', 'same', 'saw',
                     'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'sees', 'several',
                     'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since',
                     'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere',
                     'state', 'states', 'still', 'such', 'sure', 'take', 'taken', 'than', 'that', 'the', 'their',
                     'them', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this',
                     'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today',
                     'together', 'too', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under',
                     'until', 'up', 'us', 'use', 'used', 'uses', 'very', 'want', 'wanted', 'wanting', 'wants',
                     'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where',
                     'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within',
                     'without', 'work', 'worked', 'working', 'works', 'would', 'year', 'years', 'yet', 'you',
                     'young', 'younger', 'youngest', 'you', 'yours', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                     'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2',
                     '3', '4', '5', '6', '7', '8', '9', '0']
    return stopwordsList
