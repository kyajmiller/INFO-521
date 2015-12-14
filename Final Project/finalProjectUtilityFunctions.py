def makeDataSet(dataLines):
    dataSet = []

    for line in dataLines:
        state, tweet = line.split('\t', 1)


def getLabelStatePoliticalAffiliation(state):
    # if state is liberal, label = 0; if conservative, label = 1
    # declare lists of liberal states and conservative states
    liberalStates = ['WA', 'OR', 'CA', 'NV', 'CO', 'NM', 'IA', 'WI', 'MI', 'IL', 'OH', 'VA', 'DC', 'MD', 'DE', 'PA',
                     'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'VT', 'ME', 'HI', 'FL']

    conservativeStates = ['ID', 'MT', 'WY', 'UT', 'AZ', 'ND', 'SD', 'NE', 'KS', 'OK', 'TX', 'MO', 'AR', 'LA', 'IL',
                          'KY', 'TN', 'MS', 'AL', 'WV', 'NC', 'SC', 'GA', 'AK']

    if state in liberalStates:
        label = 0
    elif state in conservativeStates:
        label = 1

    return label


def checkLexicons(tokens):
    # liberal lexicon vs conservative lexicon
    liberalFoods = ['curry', 'bistro', 'fresh', 'fruit', 'strawberry', 'crunchy', 'thin', 'coconut', 'lamb', 'gnocchi',
                'fusili', 'radiatore', 'rice', 'wine', 'beer', 'diet', 'tap', 'fusion', 'vegetarian', 'foodie',
                'organic', 'seafood', 'toast', 'bagel', 'jamba', 'sbarro', 'chipotle', 'aubonpain', 'qdoba',
                'wienerschnitzel', 'starbucks', 'wingstop', 'panera', 'tacobell', 'quiznos', 'dunkin', 'donut',
                'dunkindonuts', 'pfchangs', 'cheesecakefactory', 'cpk', 'californiapizzakitchen', 'buffalowildwings',
                'ihop', 'wholefoods', 'whole foods', 'traderjoes', 'trader', "joe's", "trader joe's", 'safeway', 'frys',
                'fredmeyer', 'fred meyer', 'albertsons', 'osco', 'target', 'supertarget', 'macaroni', 'acme', 'blimpie',
                'ontheborder', 'asian']
    liberalFoodsHashtags = ['#' + l for l in liberalFoods]

    conservativeFoods = ['meatloaf', 'potato', 'bean', 'gravy', 'soda', 'mcdonalds', 'steak', 'cooked', 'grape', 'soft',
                'deep dish', 'burger', 'grill', 'tuna', 'casserole', 'meatloaf', 'linguine', 'rotini', 'spaghetti',
                'juice', 'chinese', 'cheeseburger', 'bacon', 'applebees', 'schlotzskys', 'chickfila', 'arbys', 'sonic',
                'checkers', 'hardees', 'dominos', 'mcdonalds', 'wendys', 'kfc', 'subway', 'panda', 'express',
                'pandaexpress', 'olivegarden', 'olive', 'garden', 'redlobster', 'goldencorral', 'buffet', 'hooters',
                'papamurphys', "murphy's", 'dennys', "denny's", 'krystal', 'dairy queen', 'dairyqueen', 'churchs',
                "church's", 'papajohns', "john's", 'krispykreme', 'krispy kreme', 'walmart', 'foodlion', 'food lion',
                'kroger', 'harristeeter', 'harris', 'teeter', 'publix']
    conservativeFoodsHashtags = ['#' + c for c in conservativeFoods]
