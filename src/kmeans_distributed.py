# Chuẩn bị dữ liệu, chuyển dữ liệu từ csv thành Spark DataFrame
import random

# Su dung SparkSQL
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
# Thiet lap bien moi truong
from pyspark import SparkConf
conf = SparkConf()
conf.set('spark.executor.memory', '2g')
spark = SparkSession.builder.master("local[*]").appName('kmeans').getOrCreate()
# Load dataset
df = spark.read.csv("../data/cleaned-data.csv", header=True, sep=",")

df = df.withColumn("Start_Lat",df['Start_Lat'].cast(FloatType())).withColumn("Start_Lng",df['Start_Lng'].cast(FloatType()))
# Cai dat thuat toan k-means:

# Cap nhat toa do cua center
def compute_new_centers():
    query = '''
        SELECT 
            AVG(Start_Lng) AS x, 
            AVG(Start_Lat) AS y,
            ID AS id
        FROM points
        GROUP BY id
    '''
    centers = spark.sql(query)
    centers.createOrReplaceTempView('centers')
    centers.cache()

# Ham tinh toan khoang cach giua cac diem (khong phai center) den center
def compute_distances():
    query = '''
        SELECT
            p.Start_Lng AS x,
            p.Start_Lat AS y,
            c.id AS cluster_id,
            SQRT( POWER(p.x - c.x, 2) + 
                  POWER(p.y - c.y, 2) ) AS distance
        FROM points as p
        CROSS JOIN centers as c
    '''
    distances = spark.sql(query)
    distances.createOrReplaceTempView('distances')
    distances.cache()

# Gan cac diem du lieu vao cum
def compute_new_clustering():
    query = '''
        WITH closest AS (
            SELECT
                x, y,
                MIN(distance) AS min_dist
            FROM distances
            GROUP BY x, y )
        SELECT
            sdf1.x AS x, 
            sdf1.y AS y, 
            sdf1.cluster_id AS cluster_id
        FROM distances AS sdf1
        INNER JOIN closest AS sdf2
        ON sdf1.x = sdf2.x
        AND sdf1.y = sdf2.y
        AND sdf1.distance = sdf2.min_dist
    '''    
    
    points = spark.sql(query)
    points.createOrReplaceTempView('points')
    points.cache()
    
# Vong lap cua thuat toan
def kmeans_loop():
    #B2
    compute_distances()
    compute_new_clustering()
    #B3
    compute_new_centers()

def create_centers_sdf(centers):   
    field_x = StructField('x', FloatType(), True)
    field_y = StructField('y', FloatType(), True)
    field_id = StructField('id', IntegerType(), True)
    schema = StructType([field_x, field_y, field_id])
    return spark.createDataFrame(centers, schema)


def init_centers(k=5):
    centers = []
    
    min_lat = spark.sql("SELECT MIN(Start_Lat) FROM points").collect()[0]['min(Start_Lat)']
    min_lng = spark.sql("SELECT MIN(Start_Lng) FROM points").collect()[0]['min(Start_Lng)']
    max_lat = spark.sql("SELECT MAX(Start_Lat) FROM points").collect()[0]['max(Start_Lat)']
    max_lng = spark.sql("SELECT MAX(Start_Lng) FROM points").collect()[0]['max(Start_Lng)']
    
    for i in range(k):
        x = random.uniform(min_lng, max_lng)    
        y = random.uniform(min_lat, max_lat)
        centers += {'x': x, 'y': y,'id': i}

    return create_centers_sdf(centers)
     


# Khoi tao thuat toan
def kmeans_distributed(k=5, epochs=1):
    df.createOrReplaceTempView('points')
    #B1
    centers = init_centers(k)
    centers.createOrReplaceTempView('centers')
    
    for _ in range(epochs): #for(int _ = 0; _ < epochs; _++)
        kmeans_loop()
        
# Ham tra ve dataframe la cac diem trong TempleView points
def get_spark_clustering():
    clustering = spark.sql('SELECT * FROM points')
    return clustering.toPandas()