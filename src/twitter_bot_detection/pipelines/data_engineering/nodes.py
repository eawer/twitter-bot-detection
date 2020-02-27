from pyspark import SparkContext, SQLContext, sql
from kedro.contrib.io.pyspark import SparkDataSet
from kedro.io import ParquetLocalDataSet

# sc = SparkContext()
# sqlContext = SQLContext(sc)

# tweets = sqlContext.read.json('./data/01_raw/tweets.jsonl')

# print(tweets.map(lambda x: x.user.id_str))

def convert_to_parquet(tweets: SparkDataSet) -> ParquetLocalDataSet:
    return tweets.coalesce(1)

#     return companies

