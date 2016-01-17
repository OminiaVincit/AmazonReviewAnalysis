u'''Setting for environment'''

class Settings:
    u''' For environmental settings '''

    def __init__(self):
        u'''Init'''
        pass

    DATASET_FILE = ''
    MONGO_CONNECTION_STRING = 'mongodb://localhost:27017'
    DATABASE = 'amazon'
    
    STR_DIM = 12
    TFIDF_DIM = 1024
    GALC_DIM = 39
    LIWC_DIM = 64
    INQUIRER_DIM = 182
    TOPICS_DIM = 64

    SOURCE_DIR = r'/home/zoro/work/AmazonDataset/reviews'
    PROCESSED_DIR = r'/home/zoro/work/AmazonDataset/processed_rv'

    PREFIX = 'Amazon'
    CATEGORIES = [
            'Home_and_Kitchen',
            'Sports_and_Outdoors',
            'Automotive',
            'Baby',
            'Beauty',
            'Office_Products',
            'Clothing_Shoes_and_Jewelry',
            'Toys_and_Games',
            'Musical_Instruments',
            'Pet_Supplies',
            'Health_and_Personal_Care',
            'Books', 
            'Electronics', 
            'Movies_and_TV'
            ]
    FEATURES = ['STR', 'TOPICS_64', 'TFIDF', 'LIWC', 'INQUIRER', 'GALC']
